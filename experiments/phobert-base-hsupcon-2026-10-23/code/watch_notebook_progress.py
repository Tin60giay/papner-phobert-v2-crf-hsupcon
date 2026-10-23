import sys,time,queue,os
from jupyter_client import BlockingKernelClient
from pathlib import Path
connection = sys.argv[1] if len(sys.argv)>1 else None
while not connection:
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=proc.read_bytes().decode().split('\0')
            if 'ipykernel_launcher' in argv and '-f' in argv:
                candidate=argv[argv.index('-f')+1]
                if Path(candidate).exists():connection=candidate;break
        except (OSError,UnicodeError,ValueError):pass
    if not connection:time.sleep(0.25)
log=Path('/workspace/phobert_base_runner_20261023/live.log').open('a',buffering=1)
def emit(s):
    sys.stdout.write(s);sys.stdout.flush();log.write(s);log.flush()
emit('PhoBERT base + CRF + Hierarchical SupCon | test dedup: 5162 sentences\n')
emit('Training runs on Vast in tmux; closing your PC/browser does not stop it.\n')
k=BlockingKernelClient(connection_file=connection);k.load_connection_file();k.start_channels(shell=False,stdin=False,hb=False)
last=time.monotonic()
try:
 while True:
    try:m=k.get_iopub_msg(timeout=5)
    except queue.Empty:
        if Path('/workspace/phobert_base_runner_20261023/exit_code.txt').exists():
            emit('\nNotebook exit code: '+Path('/workspace/phobert_base_runner_20261023/exit_code.txt').read_text());break
        continue
    c=m['content'];t=m['header']['msg_type']
    if t=='stream':emit(c.get('text',''))
    elif t in ('display_data','execute_result','update_display_data'):
        text=c.get('data',{}).get('text/plain','')
        if text:emit('\r'+text+'\n')
    elif t=='error':emit('\n'+'\n'.join(c.get('traceback',[]))+'\n')
    elif t=='execute_input':emit('\n--- Notebook cell execution '+str(c.get('execution_count'))+' ---\n')
finally:k.stop_channels();log.close()
