from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.database import SessionLocal
from database.models import User
from schemas.user_schema import UserCreate


router = APIRouter(
    prefix="/api/users",
    tags=["Users"]
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@router.post("/")
def create_user(user: UserCreate, db: Session = Depends(get_db)):

    new_user = User(
        name=user.name,
        age=user.age,
        language=user.language,
        interaction_mode=user.interaction_mode
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User created successfully",
        "user": {
            "id": new_user.id,
            "name": new_user.name,
            "age": new_user.age,
            "language": new_user.language,
            "interaction_mode": new_user.interaction_mode
        }
    }


@router.get("/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        return {
            "message": "User not found"
        }

    return {
        "id": user.id,
        "name": user.name,
        "age": user.age,
        "language": user.language,
        "interaction_mode": user.interaction_mode
    }