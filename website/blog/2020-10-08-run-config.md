---
slug: run-config
title: What is a Run Configuration?
author: Amin Saied
author_title: Data Scientist @ Azure ML
author_url: https://github.com/aminsaied
author_image_url: https://avatars.githubusercontent.com/aminsaied
tags: [run]
---

What _is_ a `RunConfiguration`?  In practice you should never have to build one by hand.
Azure ML provides higher-level concepts that to handle this for you - for example, the 
`ScriptRunConfig`. It is however useful to know what aspects of a run it controls, and to see how to modify
the parts you may have cause to change.

<!--truncate-->

## ScriptRunConfig

The `ScriptRunConfig` is a good place to start. The clue is in the name
here: this class combines your script together with your desired run-time configuration:

- Script:
    - Source code: usually a directory e.g. `src/` consisting of models and training code
    - Script arguments: a series of command-line arguments that control various hyperparameters
- Run Configuration: Everything you need to make your script run as intended in a remote environment.