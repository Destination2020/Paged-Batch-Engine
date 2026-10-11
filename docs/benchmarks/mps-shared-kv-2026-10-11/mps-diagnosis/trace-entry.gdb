set pagination off
set confirm off
set breakpoint pending on
set debuginfod enabled off
python
import gdb,json
seen=set()
class EntryReturn(gdb.FinishBreakpoint):
 def __init__(self,frame):super().__init__(frame,internal=True)
 def stop(self):
  print('MPS_SERVER_ENTRY_RETURN='+str(int(gdb.parse_and_eval('$eax'))),flush=True)
  return False
class ServerEntry(gdb.Breakpoint):
 def stop(self):
  print('MPS_SERVER_ENTRY='+hex(int(gdb.parse_and_eval('$pc'))),flush=True)
  print(gdb.execute('x/3wx $rdi',to_string=True),flush=True)
  EntryReturn(gdb.newest_frame())
  return False
class TableReturn(gdb.FinishBreakpoint):
 def __init__(self,frame,out,uuid):
  super().__init__(frame,internal=True);self.out,self.uuid=out,uuid
 def stop(self):
  result=int(gdb.parse_and_eval('$eax'))
  print('EXPORT_TABLE_RESULT='+str(result)+' UUID='+self.uuid,flush=True)
  if result==0 and self.uuid=='a392d3871cf47f4d84b1719b17aeac8f':
   table=int.from_bytes(gdb.selected_inferior().read_memory(self.out,8).tobytes(),'little')
   target=int.from_bytes(gdb.selected_inferior().read_memory(table+32,8).tobytes(),'little')
   if target not in seen:
    seen.add(target);ServerEntry('*'+hex(target),internal=True)
  return False
class TableEntry(gdb.Breakpoint):
 def stop(self):
  uuid=gdb.selected_inferior().read_memory(int(gdb.parse_and_eval('$rsi')),16).tobytes().hex()
  TableReturn(gdb.newest_frame(),int(gdb.parse_and_eval('$rdi')),uuid)
  return False
TableEntry('cuGetExportTable')
end
run
