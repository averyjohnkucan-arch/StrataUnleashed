"""Independent pair-codec check against the vendor's built ggml dequantizers.
Pass --vendor-library and --probe-library explicitly; no inference or downloads.
"""
import argparse,ctypes as C,json,random,struct
p=argparse.ArgumentParser();p.add_argument('--vendor-library',required=True);p.add_argument('--probe-library',required=True);a=p.parse_args()
v=C.CDLL(a.vendor_library);q=C.CDLL(a.probe_library)
q.tqp_probe.argtypes=[C.c_int,C.c_void_p,C.POINTER(C.c_float)];q.tqp_probe.restype=None
rng=random.Random(20261019);rows=[]
for k in [5,6,7]:
 ref=getattr(v,f'dequantize_row_tqp{k}');ref.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.c_int64];ref.restype=None
 count=0
 for scale in [0.,1.,-1.,.125,8.,2**-14]:
  for i in range(32):
   data=struct.pack('<e',scale)+bytes(rng.getrandbits(8) for _ in range(8*k));buf=C.create_string_buffer(data);expected=(C.c_float*128)();actual=(C.c_float*128)()
   ref(buf,expected,128);q.tqp_probe(k,buf,actual)
   assert bytes(expected)==bytes(actual),(k,scale,i,max(abs(x-y) for x,y in zip(expected,actual)))
   count+=1
 rows.append({'format':f'TQP{k}','randomBlocks':count,'values':count*128,'bitIdentical':True})
print(json.dumps({'state':'passed','cases':rows,'scope':'CPU state extraction, codebook and FP16 scaling; not GPU expert/inference parity'}))
