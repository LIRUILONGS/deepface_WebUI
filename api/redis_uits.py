import redis

class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host='192.168.26.55', port=6379, password="123456",db=db)

# 字符串
    def str_set(self, key, value):
        self.r.set(key, value)
        return self.r

    def str_get(self, key):
        return self.r.get(key)
    

    def delete(self, key):
        self.r.delete(key)
        return self.r
        
# 列表
    def lpush(self, key, value):
        self.r.lpush(key, value)
        return self.r 
        

    def rpush(self, key, value):
        self.r.rpush(key, value)
        return self.r 

    def lrange(self, key, start, end):
        return self.r.lrange(key, start, end)

# 集合
    def sadd(self, key, value):
        self.r.sadd(key, value)
        return self.r

    def smembers(self, key):
        return self.r.smembers(key)

# 哈希表
    def hset(self, key, field, value):
        self.r.hset(key, field, value)
        return self.r

    def hget(self, key, field):
        return self.r.hget(key, field)
    
    def hdel(self, key, field,boo=False):
        self.r.hdel(key, field)
        if boo:
            return self.r.hgetall(key)
        else:
            return self.r


    def hgetall(self, key):
        return self.r.hgetall(key)

# 有序集合
    def zadd(self, key, score, member):
        self.r.zadd(key, {member: score})
        return self.r 

    def zrange(self, key, start, end):
        return self.r.zrange(key, start, end)

    def zrangebyscore(self, key, min_score, max_score):
        return self.r.zrangebyscore(key, min_score, max_score)

def con_test():
    # 创建 Redis 连接对象
    r = redis.Redis(host='192.168.26.55', port=6379, password="123456", db=0)

    # 测试连接是否成功
    try:
        r.ping()
        print('连接成功')
    except redis.exceptions.ConnectionError:
        print('连接失败')


if __name__ == "__main__":
    con_test()
