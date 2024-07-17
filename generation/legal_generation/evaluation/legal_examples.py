import os
import html


class LegalExample:
    def __init__(self, root: str):
        self.save_path = os.path.join(root, 'examples')
        self.index_path = os.path.join(root, 'generation.html')
        self.example_ids = []

    def index_page(self):
        index = ['<body>\n<ul>\n']
        for did in self.example_ids:
            index.append(f'<li><a href="./examples/{did}.html">{did}</a></li>')
        index.append('\n</ul></body>')
        with open(self.index_path, 'w') as fp:
            fp.write('\n'.join(index))

    def add_example(self, did: str, gen: str, ex):
        os.makedirs(self.save_path, exist_ok=True)
        self.example_ids.append(did)

        # reference
        lines = ['<body>', '<h1>References</h1>']
        cids = []
        for citation in ex['short_citations']:
            split = citation.index('\n')
            cid = citation[:split].strip()
            cids.append(cid)
            lines.append('<h2 id="{}">{}</h2>\n{}\n'.format(
                cid.replace(' ', '.'), cid, self.html_paragraph(citation[split:]))
            )
        # previous text
        lines.append('<h1>Previous Text</h1>')
        lines.append(self.html_paragraph(ex['previous_text']))

        replacements = [(cid, '<a href="{}">{}</a>'.format('#' + cid.replace(' ', '.'), cid)) for cid in cids]
        replacements.extend([(cid, f'<span style="background-color:green">{cid}</span>') for cid in ex['hit_cites']])
        replacements.extend([(cid, f'<span style="background-color:red">{cid}</span>') for cid in ex['wrong_cites']])

        def replace_refs(text: str) -> str:
            for ele, repl in replacements:
                text = text.replace(ele, repl)
            return text

        # gold text
        lines.append('<h1>Gold Text</h1>')
        lines.append(replace_refs(self.html_paragraph(ex['gold_text'])))
        # generation
        lines.append('<h1>LLM Generation</h1>')
        lines.append(replace_refs(self.html_paragraph(gen)))

        lines.append('</body>')
        html_text = '\n'.join(lines)
        with open(os.path.join(self.save_path, f'{did}.html'), 'w') as fp:
            fp.write(html_text)

    @staticmethod
    def html_paragraph(text: str) -> str:
        while '\n\n' in text:
            text = text.replace('\n\n', '\n')
        text = text.strip()
        ret = []
        for line in text.split('\n'):
            ret.append(f'<p>{html.escape(line)}</p>')
        return '\n'.join(ret)
