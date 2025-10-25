from Database.DBConfiguration.database import SessionLocal
from Database.DBmodels.database_models import User, Team, TeamMember, JobLogs
from Database.DBmodels.extended_database_models import Role, Permission
from datetime import datetime, timedelta
import random

db = SessionLocal()
try:
    # Create test users
    test_users = [
        {"id": 1, "email": "alice@example.com", "name": "Alice Johnson"},
        {"id": 2, "email": "bob@example.com", "name": "Bob Smith"},
        {"id": 3, "email": "charlie@example.com", "name": "Charlie Brown"},
    ]

    for user_data in test_users:
        existing_user = db.query(User).filter(User.id == user_data["id"]).first()
        if not existing_user:
            user = User(
                id=user_data["id"],
                email=user_data["email"],
                name=user_data["name"],
                hashed_password="hashed_password_placeholder"
            )
            db.add(user)
            print(f"Created user: {user_data['name']}")

    # Create test team
    existing_team = db.query(Team).filter(Team.id == 10).first()
    if not existing_team:
        team = Team(
            id=10,
            name="Quantum Research Team",
            description="A team focused on quantum computing research and development",
            created_by_id=1
        )
        db.add(team)
        print("Created team: Quantum Research Team")

    # Add team members
    team_members_data = [
        {"team_id": 10, "user_id": 1, "role": "admin"},
        {"team_id": 10, "user_id": 2, "role": "member"},
        {"team_id": 10, "user_id": 3, "role": "member"},
    ]

    for member_data in team_members_data:
        existing_member = db.query(TeamMember).filter(
            TeamMember.team_id == member_data["team_id"],
            TeamMember.user_id == member_data["user_id"]
        ).first()
        if not existing_member:
            member = TeamMember(**member_data)
            db.add(member)
            print(f"Added user {member_data['user_id']} to team {member_data['team_id']} as {member_data['role']}")

    # Create test jobs for team members
    devices = ["ibm_kyoto", "ibm_osaka", "ibm_sherbrooke", "ibm_brisbane"]
    statuses = ["completed", "running", "failed", "pending"]

    for i in range(15):
        user_id = random.choice([1, 2, 3])  # Random team member
        job = JobLogs(
            jobId=f"test-job-{i+1}",
            user_id=user_id,
            Device=random.choice(devices),
            Status=random.choice(statuses),
            Shots=random.randint(1000, 10000),
            JobRasied=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
            JobCompletion=datetime.utcnow() - timedelta(days=random.randint(0, 30)) if random.choice([True, False]) else None
        )
        db.add(job)
        print(f"Created job: {job.jobId} for user {user_id}")

    db.commit()
    print("\nTest data populated successfully!")

    # Verify the data
    teams = db.query(Team).all()
    print(f"\nTotal teams: {len(teams)}")
    for team in teams:
        print(f"  Team {team.id}: {team.name}")

    members = db.query(TeamMember).all()
    print(f"Total team members: {len(members)}")
    for member in members:
        print(f"  Team {member.team_id}: User {member.user_id} ({member.role})")

    jobs = db.query(JobLogs).all()
    print(f"Total jobs: {len(jobs)}")
    for job in jobs:
        print(f"  Job {job.jobId}: User {job.user_id}, Status {job.Status}")

finally:
    db.close()
