from typing import Literal


maskKeyPointsType = Literal[
    "반팔",
    "긴팔",
    "반팔 아우터",
    "긴팔 아우터",
    "조끼",
    "슬링",
    "반바지",
    "긴바지",
    "치마",
    "반팔 원피스",
    "긴팔 원피스",
    "조끼 원피스",
    "슬링 원피스",
]
"""maskKeyPoints의 Literal 타입"""

TopMeaType = Literal["어깨너비", "가슴단면", "총장", "소매길이"]
"""긴팔 실측 타입"""


BottomMeaType = Literal["허리단면", "밑위", "엉덩이단면", "허벅지단면", "총장", "밑단단면"]
"""긴바지 실측 타입"""
