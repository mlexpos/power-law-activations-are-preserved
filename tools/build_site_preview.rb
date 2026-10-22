# Minimal stand-in for Jekyll: front matter, the few Liquid tags the site uses, kramdown (GFM, MathJax).
require 'kramdown'
require 'kramdown-parser-gfm'
require 'yaml'
require 'fileutils'
src, out = ARGV
cfg = YAML.load_file(File.join(src, '_config.yml'))
baseurl = cfg.fetch('baseurl', '').sub(%r{/$}, '')
layout = File.read(File.join(src, '_layouts', 'default.html'))
Dir.glob(File.join(src, '**', '*.md')).each do |f|
  rel = f.sub(src + '/', '')
  text = File.read(f)
  fm = {}
  if text.start_with?('---')
    _, y, text = text.split(/^---\s*$/, 3)
    fm = YAML.safe_load(y) || {}
  end
  text = text.gsub('{{ site.repo_url }}', cfg['repo_url']).gsub('{{ site.paper_url }}', cfg['paper_url'])
  html = Kramdown::Document.new(text, input: 'GFM', math_engine: 'mathjax', hard_wrap: false, auto_ids: true).to_html
  url = '/' + rel.sub(/\.md$/, '.html').sub(/(^|\/)index\.html$/, '\1')
  nav = cfg['nav'].map { |i| "<a href=\"#{baseurl}#{i['url']}\"#{url == i['url'] ? ' class="here"' : ''}>#{i['title']}</a>" }.join
  page = layout.gsub(/\{%\s*for item in site.nav\s*%\}.*?\{%\s*endfor\s*%\}/m, nav)
  page = page.gsub(/\{% if page.title %\}\{\{ page.title \}\} · \{% endif %\}/, fm['title'] ? "#{fm['title']} · " : '')
  page = page.gsub('{{ site.title }}', cfg['title']).gsub('{{ site.description }}', cfg['description'])
  page = page.gsub('{{ site.baseurl }}', baseurl).gsub('{{ site.paper_url }}', cfg['paper_url'])
  page = page.gsub('{{ site.paper_title }}', cfg['paper_title']).gsub('{{ site.paper_authors }}', cfg['paper_authors'])
  page = page.gsub('{{ site.repo_url }}', cfg['repo_url']).sub('{{ content }}', html)
  dest = File.join(out, rel.sub(/\.md$/, '.html'))
  FileUtils.mkdir_p(File.dirname(dest))
  File.write(dest, page)
end
FileUtils.cp_r(File.join(src, 'assets'), out)
