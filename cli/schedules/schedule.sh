# <create_schedule>
# This action will create related resources for a schedule. It will take dozens of seconds to complete.
az ml schedule create --file cron-schedule.yml --no-wait
# </create_schedule>

# <show_schedule>
az ml schedule show -n simple_cron_job_schedule
# </show_schedule>

# <list_schedule>
az ml schedule list
# </list_schedule>

# <update_schedule>
az ml schedule update -n simple_cron_job_schedule  --set description="new description" --no-wait
# </update_schedule>

# <disable_schedule>
az ml schedule disable -n simple_cron_job_schedule --no-wait
# </disable_schedule>

# <enable_schedule>
az ml schedule enable -n simple_cron_job_schedule --no-wait
# </enable_schedule>

# <query_triggered_jobs>
# query triggered jobs from schedule, please replace the simple_cron_job_schedule to your schedule name
az ml job list --query "[?contains(display_name,'simple_cron_schedule')]" 
# </query_triggered_jobs>

# <delete_schedule>
az ml schedule delete -n simple_cron_job_schedule
# </delete_schedule>
