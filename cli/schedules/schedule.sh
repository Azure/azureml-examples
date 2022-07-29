# <create_schedule>
az ml schedule create --file cron-schedule.yml
# </create_schedule>

# <show_schedule>
az ml schedule show -n schedule_name
# </show_schedule>

# <update_schedule>
az ml schedule update -n schedule_name  --set create_job = 'azureml:job_name'
# </update_schedule>

# <disable_schedule>
az ml schedule disable -n schedule_name
# </disable_schedule>

# <enable_schedule>
az ml schedule disable -n schedule_name
# </enable_schedule>

# <query_triggered_jobs>
# query triggered jobs from schedule, please replace the <schedule name> to real schedule name
az ml job list --query "[?contains(display_name,'<schedule_name>')]" 
# </query_triggered_jobs>