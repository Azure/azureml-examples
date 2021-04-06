# Issue support process

This is a modified version of the [Azure SDK GitHub issue support process](https://devblogs.microsoft.com/azure-sdk/github-issue-support-process).

## Labels

Issues should contain one of the following:

label|description
-|-
`example issue`|Issue with an existing example.
`example request`|Request for a new example.
`enhancement`|Suggestion to improve this repository.

This label should be assigned at issue creation. If not, and it is not immediately obvious from the issue itself, a clarification from the author will be requested.

To determine whether Azure ML or the issue author needs to respond, we will follow the same process as the `azure-sdk`:

label|description
-|-
`needs-author-feedback`|Needs additional input from the issue author (or sometimes other users).
`needs-team-attention`|Needs response from Azure ML team.

Some issues will likely reveal bugs or feature gaps in Azure ML itself. In this case, the bug or feature request label will be assigned. Many issues will just be questions about an example or Azure ML.

label|description
-|-
`bug`|Reveals a bug with Azure ML.
`feature-request`|Results in a feature request to Azure ML.
`question`|A question about an example or Azure ML.

Related issue(s) may be opened in `azure-sdk` repositories and linked.

## Response times

The Azure ML team aims to respond to all issues in this repository  labelled with `needs-team-attention` within one business day. Often, response times will be much quicker.

Actual code changes - whether to an example or Azure ML itself - take much longer. Contributions to any of the open source Microsoft repositories are welcome; be sure to check the repository's contributing guidelines!

The Azure ML team will work to fix issues with examples, depending on severity, quickly. Requests for new examples - depending on any feature gaps - will be prioritized against other work. Suggestions for repository enhancements will also be priorized against other work.

## Closing issues

An issue may be closed after it is fixed and verified or no progress can be made pending a response from the author for 14+ days.

Generally the author, a verified fix via PR, or a bot should close an issue. 

A bot will additionally label an issue with `stale` if it has been labelled with `needs-author-feedback` for 7 days. An issue labelled with `stale` will be closed following 7 days without a response.
