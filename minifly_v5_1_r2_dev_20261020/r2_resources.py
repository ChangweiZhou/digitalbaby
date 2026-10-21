"""Standard-library macOS resource admission, including child probe CPU."""
import resource,subprocess,re,time,sys
_last=0.;_available=0
def children_cpu():
    r=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime
def measurement():
    global _last,_available
    now=time.monotonic()
    if now-_last>=1:
        text=subprocess.check_output(['/usr/bin/vm_stat'],text=True)
        page=int(re.search(r'page size of (\d+) bytes',text).group(1))
        def value(name):return int(re.search(r'^'+re.escape(name)+r':\s*(\d+)',text,re.M).group(1))
        _available=page*(value('Pages free')+value('Pages inactive')+value('Pages speculative'))
        _last=now
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin':rss*=1024
    return dict(RSS_bytes=rss,available_system_bytes=_available,
                headroom_method='vm_stat (free+inactive+speculative)*page_size; no purgeable double count',RSS_method='process peak RSS')
