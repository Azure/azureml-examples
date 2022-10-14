# these components are needed for the sample under cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components

az ml component create --file ../cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components/train.yml

az ml component create --file ../cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components/score.yml

az ml component create --file ../cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components/eval.yml
