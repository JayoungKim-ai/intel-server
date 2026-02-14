from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
from dotenv import load_dotenv
load_dotenv()  
app = FastAPI()

# 허용할 프론트엔드 주소 목록
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://intel-react-kpsi.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # 특정 도메인만 허용
    allow_credentials=True,          # 모두 허용인 경우 False로 변경
    allow_methods=["*"],             # 모든 HTTP 메서드(GET, POST 등) 허용
    allow_headers=["*"],             # 모든 헤더 허용
)

@app.get("/")
def home():
    return {"message":"여기는 home입니다"}

@app.get("/about")
def about():
    return {
        "name": "김철수",
        "phone": "010-123-4567",
        "address": "서울시 종로구"
    }   

import random 

# 랜덤 명언
@app.get("/quote")
def random_quote():
    quotes = [
        "성공은 준비된 자에게 찾아온다.",
        "노력은 배신하지 않는다.",
        "오늘 걷지 않으면 내일은 뛰어야 한다."
    ]
    return {"quote":random.choice(quotes)}

# 랜덤 동물
@app.get("/animal")
def random_animal():
    characteristics = ["귀여운", "용감한", "느긋한", "쏘 쿨한"]
    animals = ["고양이", "강아지", "햄스터", "너구리"]   
    return { "characteristic" : random.choice(characteristics), 
            "animal" :random.choice(animals)} 



# 랜덤 고양이
@app.get("/random_cat")
def random_cat():
    url = "https://api.thecatapi.com/v1/images/search?limit=6"
    response = httpx.get(url)
    return response.json()

# festival
@app.get("/festival")
def get_festivals():      
    service_key = os.getenv("API_SERVICE_KEY")
    url = 'http://api.data.go.kr/openapi/tn_pubr_public_cltur_fstvl_api'
    params ={'serviceKey' : service_key, 
            'pageNo' : '1', 
            'numOfRows' : '100', 
            'type' : 'json'}
    try:
        response = httpx.get(url, params=params)
        data = response.json()    
        return data["response"]["body"]["items"]
    except KeyError:
        return {"error": "API 응답 구조가 다릅니다", "raw": data}
    except Exception as e:
        return {"error": str(e)}