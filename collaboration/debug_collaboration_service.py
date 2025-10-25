from Database.DBConfiguration.database import SessionLocal
from Database.DBmodels.database_models import JobLogs, User, TeamMember
from Database.DBmodels.extended_database_models import Role, Permission
from datetime import datetime, timedelta
from collaboration import CollaborationService

db = SessionLocal()

try:
    team_id = 10
    days = 30

    print("=== DEBUGGING COLLABORATION SERVICE ===")
    print(f"Team ID: {team_id}")
    print(f"Days: {days}")

    # Create service instance
    service = CollaborationService(db)

    # Get member IDs
    member_ids = db.query(TeamMember.user_id).filter(
        TeamMember.team_id == team_id
    ).all()
    member_ids = [mid[0] for mid in member_ids]

    print(f"Member IDs: {member_ids}")
    print(f"Member ID types: {[type(mid) for mid in member_ids]}")

    if not member_ids:
        print("❌ No members found for team")
    else:
        # Check what jobs exist for these users
        since_date = datetime.utcnow() - timedelta(days=days)
        print(f"Since date: {since_date}")

        # Query jobs directly
        all_jobs = db.query(JobLogs).filter(
            JobLogs.user_id.in_(member_ids),
            JobLogs.JobRasied >= since_date
        ).order_by(JobLogs.JobRasied.desc()).all()

        print(f"Direct query found {len(all_jobs)} jobs")

        # Show details of found jobs
        for i, job in enumerate(all_jobs[:5]):  # Show first 5
            print(f"Job {i+1}: ID={job.jobId}, UserID={job.user_id}, Status={job.Status}, Date={job.JobRasied}")

        # Now call the service method
        print("\n=== CALLING SERVICE METHOD ===")
        service_jobs = service.get_team_jobs(team_id, days)
        print(f"Service method returned {len(service_jobs)} jobs")

        # Show details of service jobs
        for i, job in enumerate(service_jobs[:5]):  # Show first 5
            print(f"Service Job {i+1}: ID={job.get('jobId')}, UserID={job.get('user_id')}, Status={job.get('Status')}")

        # Check if there are jobs with NULL user_id that might belong to team members
        print("\n=== CHECKING FOR NULL USER_ID JOBS ===")
        null_user_jobs = db.query(JobLogs).filter(
            JobLogs.user_id.is_(None),
            JobLogs.JobRasied >= since_date
        ).all()
        print(f"Found {len(null_user_jobs)} jobs with NULL user_id")

        # Check if any of these NULL user_id jobs might be from team members
        # This could be the issue - jobs exist but user_id is NULL
        if null_user_jobs:
            print("NULL user_id jobs:")
            for job in null_user_jobs[:3]:
                print(f"  Job ID: {job.jobId}, Device: {job.Device}, Date: {job.JobRasied}")

finally:
    db.close()
