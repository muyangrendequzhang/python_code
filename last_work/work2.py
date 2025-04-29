import requests
import time
from bs4 import BeautifulSoup

# 使用Session来模拟浏览器行为
session = requests.Session()

# 知乎热榜页面URL
url = 'https://www.zhihu.com/hot'

# 请求头 - 模拟正常浏览器访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1'
}

# 设置请求头
session.headers.update(headers)

try:
    # 请求知乎热榜页面
    print("正在请求知乎热榜页面...")
    response = session.get(url)
    
    # 检查响应状态
    if response.status_code == 200:
        print("成功获取知乎热榜页面!")
        
        # 使用BeautifulSoup解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找热榜条目
        hot_items = soup.select('.HotList-list .HotItem')
        
        if hot_items:
            print(f"共解析出 {len(hot_items)} 条热榜数据\n")
            print("="*50)
            print("知乎热榜TOP 20:")
            print("="*50)
            
            # 打开文件准备写入
            with open('知乎热榜.txt', 'w', encoding='utf-8') as f:
                f.write(f"知乎热榜 - 抓取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 显示并保存热榜
                for i, item in enumerate(hot_items[:20], 1):
                    # 提取标题
                    title_elem = item.select_one('.HotItem-title')
                    title = title_elem.text.strip() if title_elem else "无标题"
                    
                    # 提取链接
                    link_elem = item.select_one('.HotItem-title a')
                    link = link_elem.get('href', '') if link_elem else ""
                    if link.startswith('http'):
                        url = link
                    else:
                        url = f"https://www.zhihu.com{link}"
                    
                    # 提取热度
                    metrics_elem = item.select_one('.HotItem-metrics')
                    hot_score = metrics_elem.text.strip() if metrics_elem else "0 热度"
                    
                    # 显示到控制台
                    print(f"{i}. {title}")
                    print(f"   热度: {hot_score}")
                    print(f"   链接: {url}")
                    print("-"*50)
                    
                    # 写入到文件
                    f.write(f"{i}. {title}\n")
                    f.write(f"   热度: {hot_score}\n")
                    f.write(f"   链接: {url}\n\n")
            
            print("\n数据已保存到 '知乎热榜.txt'")
        else:
            print("未能解析出热榜条目，可能是页面结构发生变化")
            # 保存页面以便调试
            with open('zhihu_hot.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("已将页面保存至 'zhihu_hot.html' 以便调试")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"响应内容前200字符: {response.text[:200]}...")

except Exception as e:
    print(f"爬取过程中出错: {e}")