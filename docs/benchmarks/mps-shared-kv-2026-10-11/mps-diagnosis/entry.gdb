set pagination off
set confirm off
set debuginfod enabled off
break *0x40185a
commands
silent
printf "MPS_ENTRY: table=%p target=%p\n", $rax, *(void**)($rax+32)
x/12i *(void**)($rax+32)
x/5wx $rdi
continue
end
break *0x40185d
commands
silent
printf "MPS_ENTRY_RETURN: %d\n", $eax
continue
end
run
