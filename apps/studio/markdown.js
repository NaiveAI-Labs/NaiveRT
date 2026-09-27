import MarkdownIt from './vendor/markdown-it/markdown-it.mjs';

// Only parser-generated markup enters the conversation DOM. Model HTML stays literal;
// complete HTML artifacts retain their separate opaque-origin iframe preview.
const markdown = new MarkdownIt({html: false, linkify: false, typographer: false});
const linkOpen = markdown.renderer.rules.link_open;
markdown.renderer.rules.link_open = (tokens, index, options, env, self) => {
  tokens[index].attrSet('target', '_blank');
  tokens[index].attrSet('rel', 'noopener noreferrer');
  return linkOpen ? linkOpen(tokens, index, options, env, self) : self.renderToken(tokens, index, options);
};
// Model-supplied image URLs must not trigger background requests from the host page.
markdown.renderer.rules.image = (tokens, index) => markdown.utils.escapeHtml(tokens[index].content);

export function renderMarkdown(text) {
  return markdown.render(text);
}
