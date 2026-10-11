set pagination off
set confirm off
set breakpoint pending on
set debuginfod enabled off
python
import gdb,json
class Return(gdb.FinishBreakpoint):
 def __init__(self,frame,info,out):
  super().__init__(frame,internal=True)
  self.info,self.out=info,out
 def stop(self):
  self.info['return_code']=int(gdb.parse_and_eval('$eax'))
  if self.out:
   pointer=int.from_bytes(gdb.selected_inferior().read_memory(self.out,8).tobytes(),'little')
   self.info['table_pointer']=hex(pointer)
   if pointer:
    try:self.info['table_first_64_bytes']=gdb.selected_inferior().read_memory(pointer,64).tobytes().hex()
    except gdb.MemoryError:pass
  print('MPS_DIAG '+json.dumps(self.info),flush=True)
  return False
class Entry(gdb.Breakpoint):
 def stop(self):
  pc=int(gdb.parse_and_eval('$pc'))
  info={'function':self.location,'pc':hex(pc),'library':gdb.solib_name(pc)}
  out=0
  if self.location=='cuGetExportTable':
   uuid=int(gdb.parse_and_eval('$rsi'))
   out=int(gdb.parse_and_eval('$rdi'))
   info['uuid_bytes']=gdb.selected_inferior().read_memory(uuid,16).tobytes().hex()
  Return(gdb.newest_frame(),info,out)
  return False
Entry('cuGetExportTable')
Entry('cuInit')
end
run
