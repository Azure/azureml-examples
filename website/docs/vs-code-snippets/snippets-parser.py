import json
from typing import List


class Snippet:
    """Handle json snippets
    
    Parse json (VS Code) snippets file and generate markdown summary.
    """

    def __init__(self, name, snippet_json):
        self.name = name
        self.description = snippet_json.get("description")
        self.prefix = self._read_prefix(snippet_json.get("prefix"))
        self.body = snippet_json.get("body")

    def __repr__(self):
        return f"Snippet({self.name})"

    @staticmethod
    def _read_prefix(prefix):
        """Guarentee prefix is of type List."""
        if type(prefix) == list:
            return prefix
        else:
            assert type(prefix) == str
            return [prefix]

    def to_markdown(self) -> List[str]:
        """Convert snippet to markdown (as list of lines)."""
        lines = []

        # add heading
        heading = f"### {self.name}"
        lines.append(heading)
        lines.append("")

        # add description
        description = f"Description: {self.description}"
        lines.append(description)
        lines.append("")

        # add prefix(es)
        if len(self.prefix) > 1:
            prefix = f"Prefixes: "
        else:
            prefix = f"Prefix: "
        for p in self.prefix:
            prefix += f"`{p}`, "
        prefix = prefix[:-2]  # remove trailing comma and whitespace
        lines.append(prefix)
        lines.append("")

        # add python snippet
        lines.append("```python")
        for line in self.body:
            if line == "$0":
                continue
            lines.append(line)
        lines.append("```")

        return lines

    @staticmethod
    def _convert_to_json(body):
        json_body = []
        for line in body[:-1]:
            line = '"' + line + '",'
            json_body.append(line)
        line = '"' + body[-1] + '"'
        json_body.append(line)
        return json_body


frontmatter = """---
title: VS Code Snippets
description: A collection of VS Code Snippets for working with Azure ML.
---

We have compiled a collection of useful templates in the form of
[VS code snippets](https://code.visualstudio.com/docs/editor/userdefinedsnippets).

![VS Code Snippets](vs-code-snippets-demo.gif)

To add these snippets to your VS Code: `ctrl+shift+p` > Type 'Configure user
snippets' > Select `python.json`. All of these snippets are available here:
[python.json](https://github.com/Azure/azureml-examples/blob/main/website/docs/vs-code-snippets/python.json)

"""

if __name__ == "__main__":

    # parse snippets
    with open("python.json") as f:
        snippets_file = json.load(f)

    snippets = []
    for name, snippet_json in snippets_file.items():
        snippet = Snippet(name, snippet_json)
        snippets.append(snippet)

    # create file and write frontmatter
    md_filename = "snippets.md"
    with open(md_filename, "w") as f:
        # write frontmatter
        f.writelines(frontmatter)

        # write each snippet
        for snippet in snippets:
            lines = snippet.to_markdown()
            for line in lines:
                f.write(line + "\n")
