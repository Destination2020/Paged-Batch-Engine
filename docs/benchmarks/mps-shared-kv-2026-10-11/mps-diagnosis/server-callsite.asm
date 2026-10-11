
/opt/pbe-mps/565.57.01/bin/nvidia-cuda-mps-server:     file format elf64-x86-64


Disassembly of section .text:

0000000000401809 <.text+0x5e9>:
  401809:	e8 02 fa ff ff       	call   401210 <dlerror@plt>
  40180e:	be 02 00 00 00       	mov    esi,0x2
  401813:	48 8d 3d d1 18 00 00 	lea    rdi,[rip+0x18d1]        # 4030eb <dlerror@plt+0x1edb>
  40181a:	e8 e1 f8 ff ff       	call   401100 <dlopen@plt>
  40181f:	49 89 c4             	mov    r12,rax
  401822:	48 85 c0             	test   rax,rax
  401825:	74 3e                	je     401865 <dlerror@plt+0x655>
  401827:	48 8d 35 ca 18 00 00 	lea    rsi,[rip+0x18ca]        # 4030f8 <dlerror@plt+0x1ee8>
  40182e:	48 89 c7             	mov    rdi,rax
  401831:	e8 aa f9 ff ff       	call   4011e0 <dlsym@plt>
  401836:	48 85 c0             	test   rax,rax
  401839:	74 2a                	je     401865 <dlerror@plt+0x655>
  40183b:	48 8d bd 70 ff ff ff 	lea    rdi,[rbp-0x90]
  401842:	48 8d 35 e7 1a 00 00 	lea    rsi,[rip+0x1ae7]        # 403330 <dlerror@plt+0x2120>
  401849:	ff d0                	call   rax
  40184b:	85 c0                	test   eax,eax
  40184d:	75 16                	jne    401865 <dlerror@plt+0x655>
  40184f:	48 8b 85 70 ff ff ff 	mov    rax,QWORD PTR [rbp-0x90]
  401856:	48 8d 7d a4          	lea    rdi,[rbp-0x5c]
  40185a:	ff 50 20             	call   QWORD PTR [rax+0x20]
  40185d:	85 c0                	test   eax,eax
  40185f:	0f 84 06 05 00 00    	je     401d6b <dlerror@plt+0xb5b>
  401865:	b8 01 00 00 00       	mov    eax,0x1
  40186a:	e9 61 fb ff ff       	jmp    4013d0 <dlerror@plt+0x1c0>
