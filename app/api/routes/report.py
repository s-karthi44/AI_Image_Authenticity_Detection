from fastapi import APIRouter

router = APIRouter()

@router.get("/report/{id}")
async def get_report(id: str):
    # GET /report/{id}
    pass
