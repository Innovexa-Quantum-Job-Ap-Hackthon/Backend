from Database.DBConfiguration.database import SessionLocal
from Database.DBmodels.database_models import Team, TeamMember
from Database.DBmodels.extended_database_models import Role, Permission

db = SessionLocal()
try:
    # Check all teams
    all_teams = db.query(Team).all()
    print(f'Total teams: {len(all_teams)}')
    for team in all_teams:
        print(f'  Team ID: {team.id}, Name: {team.name}, Created by: {team.created_by_id}')

    # Check team members for all teams
    all_team_members = db.query(TeamMember).all()
    print(f'Total team members: {len(all_team_members)}')
    for member in all_team_members:
        print(f'  Team ID: {member.team_id}, User ID: {member.user_id}, Role: {member.role}')

    # Check team 9 specifically
    team_9 = db.query(Team).filter(Team.id == 9).first()
    if team_9:
        print(f'Team 9 details: ID={team_9.id}, Name={team_9.name}, Created by={team_9.created_by_id}')
        team_9_members = db.query(TeamMember).filter(TeamMember.team_id == 9).all()
        print(f'Team 9 members: {len(team_9_members)}')
        for member in team_9_members:
            print(f'  User ID: {member.user_id}, Role: {member.role}')
    else:
        print('Team 9 not found')

finally:
    db.close()
