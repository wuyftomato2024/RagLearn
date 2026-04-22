from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL ="mysql+pymysql://root:tomato123@localhost/test_db"

# 连接数据库用的东西,Python 这边通往 MySQL 的接口
engine = create_engine(DATABASE_URL)

# 操作数据库时临时拿来用的通道
SessionLocal = sessionmaker(
    autocommit = False ,
    autoflush = False,
    bind = engine 
)

# 负责定义数据库表模型
Base = declarative_base()