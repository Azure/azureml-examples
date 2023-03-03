
|DELETE | replaced by notebook in get-started-notebooks |
|---------|---------|
|azurem-getting-started-studio | quickstart |
|azureml-in-a-day | quickstart |
|e2d-ds-experience | pipeline |

BEFORE MERGING, NEED TO:
1. Switch docs refs on release branch to the new-tutorial-series branch
1. Switch docs refs on MAIN also to the new-tutorial-series branch (so links don't break when we merge)
1. Find out how link to azureml-getting-started-studio is created.  Needs to be updated to quickstart notebook.
1. Once merged to main, change docs to link to main again.