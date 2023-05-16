import hashlib


def get_table_name(prefix, job_id):
    # h = hashlib.md5(f"torch{job_id}".encode()).hexdigest()
    # return f"torch{h}"
    # clean job_id to be a valid table name (alphanumeric, lowercase)
    job_id = "".join([c for c in job_id if c.isalnum()]).lower()
    return f"{prefix}{job_id}"
