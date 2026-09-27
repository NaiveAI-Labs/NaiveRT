// Queue-only feasibility screen. prepared_q_tail excludes all q preparation;
// this is an optimistic residual-cost measurement, not measured target overlap.
#define main sharded_benchmark_main
#include "bench_sharded.cu"
#undef main
#include "split_q.cuh"
#include <array>
static_assert(sizeof(shard::Args)==sizeof(split_shard::Args));
static_assert(sizeof(shard::Scratch)==sizeof(split_shard::Scratch));
static_assert(sizeof(shard::Mailbox)==sizeof(split_shard::Mailbox));

int main(int argc, char** argv) {
  REQUIRE(argc>=2);
  const int repeats=argc>2?atoi(argv[2]):32, batch=16;
  REQUIRE(repeats>0);
  const std::string dir=argv[1];
  auto real_p=read<float>(dir+"/target.bin",8*V), real_q=read<float>(dir+"/draft.bin",7*V);
  auto real_ids=read<int64_t>(dir+"/proposals.bin",7);
  Rank* ranks[8];
  unsigned long long* qflags[8];
  for(int r=0;r<8;++r) {
    CK(cudaSetDevice(r));
    for(int peer=0;peer<8;++peer) if(peer!=r) {
      auto e=cudaDeviceEnablePeerAccess(peer,0);
      REQUIRE(e==cudaSuccess||e==cudaErrorPeerAccessAlreadyEnabled);cudaGetLastError();
    }
    ranks[r]=new Rank; ranks[r]->a.rank=r;
    alloc(qflags[r],9);CK(cudaMemset(qflags[r],0,9*sizeof(*qflags[r])));
  }
  split_shard::QFlags qf{};
  for(int r=0;r<8;++r) {
    qf.rank[r]=qflags[r];
    for(int peer=0;peer<8;++peer) {
      ranks[r]->a.peers[peer]=ranks[peer]->mailbox;
      ranks[r]->a.flags[peer]=ranks[peer]->flags;
    }
  }
  auto sync=[&] {for(int r=0;r<8;++r){CK(cudaSetDevice(r));CK(cudaStreamSynchronize(ranks[r]->stream));}};
  auto launch=[&](int r,int impl) {
    auto& x=*ranks[r];
    split_shard::Args a;std::memcpy(&a,&x.a,sizeof(a));
    auto* s=reinterpret_cast<split_shard::Scratch*>(x.scratch);
    if(impl==0)shard::run(x.a,x.scratch,false,x.stream);
    if(impl==1)split_shard::run_target(a,s,false,x.stream);
    if(impl==2||impl==3)split_shard::prepare_q(a,s,qf,x.stream);
    if(impl==2)split_shard::run_target(a,s,false,x.stream);
    CK(cudaGetLastError());
  };
  auto run=[&](int impl) {for(int r=0;r<8;++r){CK(cudaSetDevice(r));launch(r,impl);}sync();};
  auto input=[&](const std::vector<float>& p,const std::vector<float>& q,
                 const std::vector<int64_t>& ids,int64_t seed,int64_t counter) {
    for(int r=0;r<8;++r) {
      CK(cudaSetDevice(r));auto& x=*ranks[r];
      std::vector<float> sp(8*SHARD),sq(7*SHARD);
      for(int row=0;row<8;++row)std::copy_n(p.data()+row*V+r*SHARD,SHARD,sp.data()+row*SHARD);
      for(int row=0;row<7;++row)std::copy_n(q.data()+row*V+r*SHARD,SHARD,sq.data()+row*SHARD);
      int64_t state[]={seed,counter};
      CK(cudaMemcpy(x.input_p,sp.data(),sp.size()*4,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(x.input_q,sq.data(),sq.size()*4,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(x.ids,ids.data(),ids.size()*8,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(x.rng,state,sizeof(state),cudaMemcpyHostToDevice));
      CK(cudaMemset(x.a.watchdog,0,4));
    }
  };
  struct Output {std::vector<float> p,q;std::vector<int32_t> tokens;};
  auto snapshot=[&] {
    std::array<Output,8> o;
    for(int r=0;r<8;++r) {
      CK(cudaSetDevice(r));auto& x=*ranks[r];
      REQUIRE(download(x.a.watchdog,1)[0]==0);
      o[r]={download(x.a.p,8*SHARD),download(x.a.q,7*SHARD),download(x.a.tokens,8)};
      if(r)REQUIRE(o[r].tokens==o[0].tokens);
    }
    return o;
  };
  size_t checked=0;
  auto compare=[&](const auto& actual,const auto& expected) {
    for(int r=0;r<8;++r) {
      REQUIRE(actual[r].tokens==expected[r].tokens);
      REQUIRE(std::memcmp(actual[r].p.data(),expected[r].p.data(),8*SHARD*4)==0);
      REQUIRE(std::memcmp(actual[r].q.data(),expected[r].q.data(),7*SHARD*4)==0);
      checked+=15*SHARD;
    }
  };
  int cases=0;
  std::mt19937 gen(42);std::normal_distribution<float> normal(0,4);
  for(int kind=0;kind<11;++kind)for(int trial=0;trial<4;++trial) {
    auto p=real_p,q=real_q;auto ids=real_ids;
    if(kind==1){std::fill(p.begin(),p.end(),0);std::fill(q.begin(),q.end(),0);}
    if(kind==2){for(auto& v:p)v=normal(gen);for(auto& v:q)v=normal(gen);for(auto& id:ids)id=gen()%V;}
    if(kind>=3) {
      int accept=kind-3;
      std::fill(p.begin(),p.end(),-10000);std::fill(q.begin(),q.end(),-10000);
      for(int row=0;row<7;++row){ids[row]=row*SHARD;q[row*V+ids[row]]=0;p[row*V+ids[row]+(row<accept?0:1)]=0;}
      p[7*V+V-1]=0;
    }
    input(p,q,ids,trial==3?INT64_MAX:1234+trial,trial==3?4294967295ll:trial*7);
    run(0);auto expected=snapshot();
    for(int r=0;r<8;++r){CK(cudaSetDevice(r));CK(cudaMemset(ranks[r]->a.q,0xff,7*SHARD*4));}
    run(3);
    for(int r=0;r<8;++r) {
      CK(cudaSetDevice(r));auto qnow=download(ranks[r]->a.q,7*SHARD);
      REQUIRE(std::memcmp(qnow.data(),expected[r].q.data(),7*SHARD*4)==0);checked+=7*SHARD;
    }
    run(1);compare(snapshot(),expected);
    run(2);compare(snapshot(),expected);
    ++cases;
  }
  fprintf(stderr,"bitwise gate passed: %d cases, %zu probability values\n",cases,checked);
  input(real_p,real_q,real_ids,1234,0);run(0);auto expected=snapshot();run(3);
  cudaGraph_t graph[8][4];cudaGraphExec_t exec[8][4];
  size_t nodes[4]={};
  for(int r=0;r<8;++r) {
    CK(cudaSetDevice(r));auto& x=*ranks[r];
    for(int impl=0;impl<4;++impl) {
      CK(cudaStreamBeginCapture(x.stream,cudaStreamCaptureModeGlobal));
      for(int i=0;i<batch;++i)launch(r,impl);
      CK(cudaStreamEndCapture(x.stream,&graph[r][impl]));
      size_t count=0;CK(cudaGraphGetNodes(graph[r][impl],nullptr,&count));
      if(r==0)nodes[impl]=count/batch;else REQUIRE(count/batch==nodes[impl]);
      REQUIRE(count%batch==0);
      CK(cudaGraphInstantiateWithFlags(&exec[r][impl],graph[r][impl],cudaGraphInstantiateFlagDeviceLaunch));
    }
  }
  auto call=[&](int impl) {
    for(int r=0;r<8;++r) {
      CK(cudaSetDevice(r));auto& x=*ranks[r];
      CK(cudaEventRecord(x.start,x.stream));CK(cudaGraphLaunch(exec[r][impl],x.stream));CK(cudaEventRecord(x.end,x.stream));
    }
    sync();float maximum=0;
    for(int r=0;r<8;++r) {
      CK(cudaSetDevice(r));auto& x=*ranks[r];float ms;
      CK(cudaEventElapsedTime(&ms,x.start,x.end));maximum=std::max(maximum,ms*1000/batch);
      REQUIRE(download(x.a.watchdog,1)[0]==0);
    }
    compare(snapshot(),expected);
    return maximum;
  };
  const char* names[]={"baseline_joint","prepared_q_tail","split_sequential","q_prepare_only"};
  printf("{\"pass_checks\":true,\"cases\":%d,\"initial_probability_values\":%zu,\"batch\":%d,\"timing\":[",cases,checked,batch);
  for(int impl=1;impl<4;++impl) {
    std::vector<std::array<float,4>> records;
    for(int rep=0;rep<repeats+4;++rep) {
      std::array<float,4> t={call(0),call(impl),call(impl),call(0)};
      if(rep>=4)records.push_back(t);
    }
    printf("%s{\"variant\":\"%s\",\"nodes_per_call\":%zu,\"baseline_nodes\":%zu,\"max_rank_abba_us\":[",impl==1?"":",",names[impl],nodes[impl],nodes[0]);
    for(size_t i=0;i<records.size();++i){auto t=records[i];printf("%s[%.6f,%.6f,%.6f,%.6f]",i?",":"",t[0],t[1],t[2],t[3]);}
    printf("]}");fflush(stdout);
  }
  printf("],\"probability_values_compared\":%zu,\"bitwise\":true}\n",checked);
}
