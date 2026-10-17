from enum import StrEnum

from pydantic import BaseModel, Field


class QueryIntent(StrEnum):
    BOOK_RECOMMEND = "book_recommend"
    BOOK_DETAIL = "book_detail"
    CONTENT_SEARCH = "content_search"
    NOTE_QUERY = "note_query"
    BOOK_COUNT = "book_count"
    GENERAL_QA = "general_qa"

class QueryRequest(BaseModel):
    query:str
    session_id:str
    is_stream:bool = False
    web_search_enabled: bool = False


class BookRecommendRequest(BaseModel):
    query: str = Field(min_length=1)
    book_name: str | None = None
    category: str | None = None
    scene_tag: str | None = None
    author_name: str | None = None
    session_id: str | None = None
    is_stream: bool = False
    web_search_enabled: bool = False


class BookRecommendResponse(BaseModel):
    session_id: str
    answer: str = ""
    recommendations: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BookDetailRequest(BaseModel):
    book_name: str = Field(min_length=1)
    session_id: str | None = None
    is_stream: bool = False
    web_search_enabled: bool = False


class BookDetailResponse(BaseModel):
    session_id: str
    book_name: str
    answer: str = ""
    details: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class QueryAsyncResponse(BaseModel):
    message:str
    session_id:str
    stream_id: str

class QueryResponse(BaseModel):
    message: str
    session_id: str
    answer:str
    done_list:list[str]
    image_urls:list[str]
    citations: list[dict] = Field(default_factory=list)
    intent: str = ""
    warnings: list[str] = Field(default_factory=list)


class HistoryDeleteResponse(BaseModel):
    message:str
    deleted_count:int

class HistoryItemResponse(BaseModel):
    id:str
    session_id:str
    role:str
    text:str
    rewritten_query:str
    item_names:list[str]
    image_urls:list[str]
    ts:float   # Any
    citations: list[dict] = Field(default_factory=list)
    intent: str = ""


class HistoryResponse(BaseModel):
    session_id:str
    items:list[HistoryItemResponse]