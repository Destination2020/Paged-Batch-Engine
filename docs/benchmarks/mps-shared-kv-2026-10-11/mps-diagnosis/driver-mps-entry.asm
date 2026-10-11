
/usr/lib/wsl/drivers/nvlt.inf_amd64_3ab002af0a386850/libcuda.so.1.1:     file format elf64-x86-64


Disassembly of section .text:

0000000000321b90 <cuStreamGetCaptureInfo_v2@@Base+0x38620>:
  321b90:	55                   	push   rbp
  321b91:	48 89 e5             	mov    rbp,rsp
  321b94:	41 54                	push   r12
  321b96:	53                   	push   rbx
  321b97:	48 83 ec 10          	sub    rsp,0x10
  321b9b:	e8 50 b2 e7 ff       	call   19cdf0 <cudbgReportDriverApiError@@Base+0x30c40>
  321ba0:	e8 cb 97 e1 ff       	call   13b370 <getpid@plt>
  321ba5:	89 c3                	mov    ebx,eax
  321ba7:	87 05 0b 02 1a 01    	xchg   DWORD PTR [rip+0x11a020b],eax        # 14c1db8 <cuMemGetAttribute@@Base+0xf9ad78>
  321bad:	39 d8                	cmp    eax,ebx
  321baf:	75 3f                	jne    321bf0 <cuStreamGetCaptureInfo_v2@@Base+0x38680>
  321bb1:	8b 05 05 02 1a 01    	mov    eax,DWORD PTR [rip+0x11a0205]        # 14c1dbc <cuMemGetAttribute@@Base+0xf9ad7c>
  321bb7:	39 d8                	cmp    eax,ebx
  321bb9:	74 14                	je     321bcf <cuStreamGetCaptureInfo_v2@@Base+0x3865f>
  321bbb:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]
  321bc0:	e8 2b 98 e1 ff       	call   13b3f0 <sched_yield@plt>
  321bc5:	8b 05 f1 01 1a 01    	mov    eax,DWORD PTR [rip+0x11a01f1]        # 14c1dbc <cuMemGetAttribute@@Base+0xf9ad7c>
  321bcb:	39 d8                	cmp    eax,ebx
  321bcd:	75 f1                	jne    321bc0 <cuStreamGetCaptureInfo_v2@@Base+0x38650>
  321bcf:	80 0d c6 39 1b 01 01 	or     BYTE PTR [rip+0x11b39c6],0x1        # 14d559c <cudbgEnableLaunchBlocking@@Base+0x2b44>
  321bd6:	48 83 c4 10          	add    rsp,0x10
  321bda:	b8 e7 03 00 00       	mov    eax,0x3e7
  321bdf:	5b                   	pop    rbx
  321be0:	41 5c                	pop    r12
  321be2:	5d                   	pop    rbp
  321be3:	c3                   	ret
  321be4:	66 90                	xchg   ax,ax
  321be6:	66 2e 0f 1f 84 00 00 	cs nop WORD PTR [rax+rax*1+0x0]
  321bed:	00 00 00
  321bf0:	4c 8d 65 ec          	lea    r12,[rbp-0x14]
  321bf4:	4c 89 e7             	mov    rdi,r12
  321bf7:	e8 54 98 e1 ff       	call   13b450 <pthread_mutexattr_init@plt>
  321bfc:	85 c0                	test   eax,eax
  321bfe:	74 10                	je     321c10 <cuStreamGetCaptureInfo_v2@@Base+0x386a0>
  321c00:	89 1d b6 01 1a 01    	mov    DWORD PTR [rip+0x11a01b6],ebx        # 14c1dbc <cuMemGetAttribute@@Base+0xf9ad7c>
  321c06:	eb c7                	jmp    321bcf <cuStreamGetCaptureInfo_v2@@Base+0x3865f>
  321c08:	0f 1f 84 00 00 00 00 	nop    DWORD PTR [rax+rax*1+0x0]
  321c0f:	00
  321c10:	be 01 00 00 00       	mov    esi,0x1
  321c15:	4c 89 e7             	mov    rdi,r12
  321c18:	e8 f3 94 e1 ff       	call   13b110 <pthread_mutexattr_settype@plt>
  321c1d:	85 c0                	test   eax,eax
  321c1f:	75 df                	jne    321c00 <cuStreamGetCaptureInfo_v2@@Base+0x38690>
  321c21:	31 f6                	xor    esi,esi
  321c23:	4c 89 e7             	mov    rdi,r12
  321c26:	e8 b5 93 e1 ff       	call   13afe0 <pthread_mutexattr_setpshared@plt>
  321c2b:	85 c0                	test   eax,eax
  321c2d:	75 d1                	jne    321c00 <cuStreamGetCaptureInfo_v2@@Base+0x38690>
  321c2f:	48 8d 3d 4a 1e 1b 01 	lea    rdi,[rip+0x11b1e4a]        # 14d3a80 <cudbgEnableLaunchBlocking@@Base+0x1028>
  321c36:	4c 89 e6             	mov    rsi,r12
  321c39:	e8 a2 8e e1 ff       	call   13aae0 <pthread_mutex_init@plt>
  321c3e:	85 c0                	test   eax,eax
  321c40:	75 be                	jne    321c00 <cuStreamGetCaptureInfo_v2@@Base+0x38690>
  321c42:	4c 89 e7             	mov    rdi,r12
  321c45:	e8 56 8d e1 ff       	call   13a9a0 <pthread_mutexattr_destroy@plt>
  321c4a:	eb b4                	jmp    321c00 <cuStreamGetCaptureInfo_v2@@Base+0x38690>
