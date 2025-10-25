from Database.DBConfiguration.database import SessionLocal
from Database.DBmodels.database_models import Team, TeamMember, JobLogs
from Database.DBmodels.extended_database_models import Role, Permission

db = SessionLocal()
try:
    # Check all teams
    all_teams = db.query(Team).all()
    print(f'Total teams: {len(all_teams)}')
    for team in all_teams:
        print(f'  Team ID: {team.id}, Name: {team.name}, Created by: {team.created_by_id}')

    # Check all team members
    all_team_members = db.query(TeamMember).all()
    print(f'Total team members: {len(all_team_members)}')
    for member in all_team_members:
        print(f'  Team ID: {member.team_id}, User ID: {member.user_id}, Role: {member.role}')

    # Check all jobs
    all_jobs = db.query(JobLogs).all()
    print(f'Total jobs: {len(all_jobs)}')
    for job in all_jobs:
        print(f'  Job ID: {job.jobId}, User ID: {job.user_id}, Device: {job.Device}, Status: {job.Status}')

finally:
    db.close()
