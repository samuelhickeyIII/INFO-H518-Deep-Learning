ΚΥ
γ³
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
₯
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8Ί―

RMSprop/dense_22/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/dense_22/bias/rms

-RMSprop/dense_22/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_22/bias/rms*
_output_shapes
:*
dtype0

RMSprop/dense_22/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameRMSprop/dense_22/kernel/rms

/RMSprop/dense_22/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_22/kernel/rms*
_output_shapes

:*
dtype0

RMSprop/dense_21/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/dense_21/bias/rms

-RMSprop/dense_21/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_21/bias/rms*
_output_shapes
:*
dtype0

RMSprop/dense_21/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *,
shared_nameRMSprop/dense_21/kernel/rms

/RMSprop/dense_21/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_21/kernel/rms*
_output_shapes

: *
dtype0
€
#RMSprop/embedding_11/embeddings/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
θΰ *4
shared_name%#RMSprop/embedding_11/embeddings/rms

7RMSprop/embedding_11/embeddings/rms/Read/ReadVariableOpReadVariableOp#RMSprop/embedding_11/embeddings/rms* 
_output_shapes
:
θΰ *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:*
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

: *
dtype0

embedding_11/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
θΰ *(
shared_nameembedding_11/embeddings

+embedding_11/embeddings/Read/ReadVariableOpReadVariableOpembedding_11/embeddings* 
_output_shapes
:
θΰ *
dtype0

"serving_default_embedding_11_inputPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
£
StatefulPartitionedCallStatefulPartitionedCall"serving_default_embedding_11_inputembedding_11/embeddingsdense_21/kerneldense_21/biasdense_22/kerneldense_22/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_519649

NoOpNoOp
―+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*κ*
valueΰ*Bέ* BΦ*
Ξ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
 
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias*
¦
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias*
'
0
!1
"2
)3
*4*
'
0
!1
"2
)3
*4*
	
+0* 
°
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
1trace_0
2trace_1
3trace_2
4trace_3* 
6
5trace_0
6trace_1
7trace_2
8trace_3* 
* 
z
9iter
	:decay
;learning_rate
<momentum
=rho	rmsg	!rmsh	"rmsi	)rmsj	*rmsk*

>serving_default* 

0*

0*
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Dtrace_0* 

Etrace_0* 
ke
VARIABLE_VALUEembedding_11/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ktrace_0* 

Ltrace_0* 

!0
"1*

!0
"1*
	
+0* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

Rtrace_0* 

Strace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

Ytrace_0* 

Ztrace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

[trace_0* 
* 
 
0
1
2
3*

\0
]1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
+0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
^	variables
_	keras_api
	`total
	acount*
H
b	variables
c	keras_api
	dtotal
	ecount
f
_fn_kwargs*

`0
a1*

^	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

d0
e1*

b	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUE#RMSprop/embedding_11/embeddings/rmsXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_21/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_21/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_22/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_22/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+embedding_11/embeddings/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7RMSprop/embedding_11/embeddings/rms/Read/ReadVariableOp/RMSprop/dense_21/kernel/rms/Read/ReadVariableOp-RMSprop/dense_21/bias/rms/Read/ReadVariableOp/RMSprop/dense_22/kernel/rms/Read/ReadVariableOp-RMSprop/dense_22/bias/rms/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_520012

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_11/embeddingsdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototal_1count_1totalcount#RMSprop/embedding_11/embeddings/rmsRMSprop/dense_21/kernel/rmsRMSprop/dense_21/bias/rmsRMSprop/dense_22/kernel/rmsRMSprop/dense_22/bias/rms*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_520079ώΝ
ζ	
§
H__inference_embedding_11_layer_call_and_return_conditional_losses_519772

inputs+
embedding_lookup_519766:
θΰ 
identity’embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????Δ
embedding_lookupResourceGatherembedding_lookup_519766Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/519766*4
_output_shapes"
 :?????????????????? *
dtype0«
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/519766*4
_output_shapes"
 :?????????????????? 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? 
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs

Ά
I__inference_sequential_11_layer_call_and_return_conditional_losses_519550

inputs'
embedding_11_519531:
θΰ !
dense_21_519535: 
dense_21_519537:!
dense_22_519540:
dense_22_519542:
identity’ dense_21/StatefulPartitionedCall’1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp’ dense_22/StatefulPartitionedCall’$embedding_11/StatefulPartitionedCallω
$embedding_11/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_11_519531*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_embedding_11_layer_call_and_return_conditional_losses_519416
+global_average_pooling1d_10/PartitionedCallPartitionedCall-embedding_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519396‘
 dense_21/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_10/PartitionedCall:output:0dense_21_519535dense_21_519537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_519443
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_519540dense_22_519542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_519459
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_519535*
_output_shapes

: *
dtype0
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£< 
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????η
NoOpNoOp!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_22/StatefulPartitionedCall%^embedding_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
©
¨
#__inference_internal_grad_fn_519973
result_grads_0
result_grads_1#
mul_sequential_11_dense_21_beta&
"mul_sequential_11_dense_21_biasadd
identity
mulMulmul_sequential_11_dense_21_beta"mul_sequential_11_dense_21_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????
mul_1Mulmul_sequential_11_dense_21_beta"mul_sequential_11_dense_21_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
Ώ
Β
I__inference_sequential_11_layer_call_and_return_conditional_losses_519600
embedding_11_input'
embedding_11_519581:
θΰ !
dense_21_519585: 
dense_21_519587:!
dense_22_519590:
dense_22_519592:
identity’ dense_21/StatefulPartitionedCall’1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp’ dense_22/StatefulPartitionedCall’$embedding_11/StatefulPartitionedCall
$embedding_11/StatefulPartitionedCallStatefulPartitionedCallembedding_11_inputembedding_11_519581*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_embedding_11_layer_call_and_return_conditional_losses_519416
+global_average_pooling1d_10/PartitionedCallPartitionedCall-embedding_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519396‘
 dense_21/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_10/PartitionedCall:output:0dense_21_519585dense_21_519587*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_519443
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_519590dense_22_519592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_519459
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_519585*
_output_shapes

: *
dtype0
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£< 
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????η
NoOpNoOp!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_22/StatefulPartitionedCall%^embedding_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_11_input
Ώ
Β
I__inference_sequential_11_layer_call_and_return_conditional_losses_519622
embedding_11_input'
embedding_11_519603:
θΰ !
dense_21_519607: 
dense_21_519609:!
dense_22_519612:
dense_22_519614:
identity’ dense_21/StatefulPartitionedCall’1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp’ dense_22/StatefulPartitionedCall’$embedding_11/StatefulPartitionedCall
$embedding_11/StatefulPartitionedCallStatefulPartitionedCallembedding_11_inputembedding_11_519603*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_embedding_11_layer_call_and_return_conditional_losses_519416
+global_average_pooling1d_10/PartitionedCallPartitionedCall-embedding_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519396‘
 dense_21/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_10/PartitionedCall:output:0dense_21_519607dense_21_519609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_519443
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_519612dense_22_519614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_519459
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_519607*
_output_shapes

: *
dtype0
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£< 
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????η
NoOpNoOp!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_22/StatefulPartitionedCall%^embedding_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_11_input

s
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519396

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ζ	
§
H__inference_embedding_11_layer_call_and_return_conditional_losses_519416

inputs+
embedding_lookup_519410:
θΰ 
identity’embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????Δ
embedding_lookupResourceGatherembedding_lookup_519410Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/519410*4
_output_shapes"
 :?????????????????? *
dtype0«
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/519410*4
_output_shapes"
 :?????????????????? 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? 
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
	
ό
.__inference_sequential_11_layer_call_fn_519578
embedding_11_input
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_519550o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_11_input

X
<__inference_global_average_pooling1d_10_layer_call_fn_519777

inputs
identityΞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519396i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Σ

#__inference_internal_grad_fn_519955
result_grads_0
result_grads_1
mul_dense_21_beta
mul_dense_21_biasadd
identityv
mulMulmul_dense_21_betamul_dense_21_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????g
mul_1Mulmul_dense_21_betamul_dense_21_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
Η	
υ
D__inference_dense_22_layer_call_and_return_conditional_losses_519459

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
―(
β
I__inference_sequential_11_layer_call_and_return_conditional_losses_519719

inputs8
$embedding_11_embedding_lookup_519687:
θΰ 9
'dense_21_matmul_readvariableop_resource: 6
(dense_21_biasadd_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:6
(dense_22_biasadd_readvariableop_resource:
identity’dense_21/BiasAdd/ReadVariableOp’dense_21/MatMul/ReadVariableOp’1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp’dense_22/BiasAdd/ReadVariableOp’dense_22/MatMul/ReadVariableOp’embedding_11/embedding_lookupk
embedding_11/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????ψ
embedding_11/embedding_lookupResourceGather$embedding_11_embedding_lookup_519687embedding_11/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding_11/embedding_lookup/519687*4
_output_shapes"
 :?????????????????? *
dtype0?
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_11/embedding_lookup/519687*4
_output_shapes"
 :?????????????????? €
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? t
2global_average_pooling1d_10/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Κ
 global_average_pooling1d_10/MeanMean1embedding_11/embedding_lookup/Identity_1:output:0;global_average_pooling1d_10/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_21/MatMulMatMul)global_average_pooling1d_10/Mean:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????R
dense_21/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_21/mulMuldense_21/beta:output:0dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
dense_21/SigmoidSigmoiddense_21/mul:z:0*
T0*'
_output_shapes
:?????????x
dense_21/mul_1Muldense_21/BiasAdd:output:0dense_21/Sigmoid:y:0*
T0*'
_output_shapes
:?????????c
dense_21/IdentityIdentitydense_21/mul_1:z:0*
T0*'
_output_shapes
:?????????Ε
dense_21/IdentityN	IdentityNdense_21/mul_1:z:0dense_21/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-519701*:
_output_shapes(
&:?????????:?????????
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_22/MatMulMatMuldense_21/IdentityN:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£< 
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:????????? 
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp^embedding_11/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2>
embedding_11/embedding_lookupembedding_11/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs

z
#__inference_internal_grad_fn_519919
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
―(
β
I__inference_sequential_11_layer_call_and_return_conditional_losses_519755

inputs8
$embedding_11_embedding_lookup_519723:
θΰ 9
'dense_21_matmul_readvariableop_resource: 6
(dense_21_biasadd_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:6
(dense_22_biasadd_readvariableop_resource:
identity’dense_21/BiasAdd/ReadVariableOp’dense_21/MatMul/ReadVariableOp’1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp’dense_22/BiasAdd/ReadVariableOp’dense_22/MatMul/ReadVariableOp’embedding_11/embedding_lookupk
embedding_11/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????ψ
embedding_11/embedding_lookupResourceGather$embedding_11_embedding_lookup_519723embedding_11/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding_11/embedding_lookup/519723*4
_output_shapes"
 :?????????????????? *
dtype0?
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_11/embedding_lookup/519723*4
_output_shapes"
 :?????????????????? €
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? t
2global_average_pooling1d_10/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Κ
 global_average_pooling1d_10/MeanMean1embedding_11/embedding_lookup/Identity_1:output:0;global_average_pooling1d_10/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_21/MatMulMatMul)global_average_pooling1d_10/Mean:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????R
dense_21/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_21/mulMuldense_21/beta:output:0dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
dense_21/SigmoidSigmoiddense_21/mul:z:0*
T0*'
_output_shapes
:?????????x
dense_21/mul_1Muldense_21/BiasAdd:output:0dense_21/Sigmoid:y:0*
T0*'
_output_shapes
:?????????c
dense_21/IdentityIdentitydense_21/mul_1:z:0*
T0*'
_output_shapes
:?????????Ε
dense_21/IdentityN	IdentityNdense_21/mul_1:z:0dense_21/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-519737*:
_output_shapes(
&:?????????:?????????
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_22/MatMulMatMuldense_21/IdentityN:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£< 
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:????????? 
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp^embedding_11/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2>
embedding_11/embedding_lookupembedding_11/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
Η	
υ
D__inference_dense_22_layer_call_and_return_conditional_losses_519833

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Η.

__inference__traced_save_520012
file_prefix6
2savev2_embedding_11_embeddings_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopB
>savev2_rmsprop_embedding_11_embeddings_rms_read_readvariableop:
6savev2_rmsprop_dense_21_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_21_bias_rms_read_readvariableop:
6savev2_rmsprop_dense_22_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_22_bias_rms_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*΅	
value«	B¨	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B ₯
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_embedding_11_embeddings_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_rmsprop_embedding_11_embeddings_rms_read_readvariableop6savev2_rmsprop_dense_21_kernel_rms_read_readvariableop4savev2_rmsprop_dense_21_bias_rms_read_readvariableop6savev2_rmsprop_dense_22_kernel_rms_read_readvariableop4savev2_rmsprop_dense_22_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesp
n: :
θΰ : :::: : : : : : : : : :
θΰ : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
θΰ :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
θΰ :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
 	
±
__inference_loss_fn_0_519842L
:dense_21_kernel_regularizer_l2loss_readvariableop_resource: 
identity’1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp¬
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_21_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

: *
dtype0
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£< 
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_21/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp
ηM
²
"__inference__traced_restore_520079
file_prefix<
(assignvariableop_embedding_11_embeddings:
θΰ 4
"assignvariableop_1_dense_21_kernel: .
 assignvariableop_2_dense_21_bias:4
"assignvariableop_3_dense_22_kernel:.
 assignvariableop_4_dense_22_bias:)
assignvariableop_5_rmsprop_iter:	 *
 assignvariableop_6_rmsprop_decay: 2
(assignvariableop_7_rmsprop_learning_rate: -
#assignvariableop_8_rmsprop_momentum: (
assignvariableop_9_rmsprop_rho: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: K
7assignvariableop_14_rmsprop_embedding_11_embeddings_rms:
θΰ A
/assignvariableop_15_rmsprop_dense_21_kernel_rms: ;
-assignvariableop_16_rmsprop_dense_21_bias_rms:A
/assignvariableop_17_rmsprop_dense_22_kernel_rms:;
-assignvariableop_18_rmsprop_dense_22_bias_rms:
identity_20’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_2’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*΅	
value«	B¨	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp(assignvariableop_embedding_11_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_21_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_21_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_22_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_22_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_rmsprop_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_rmsprop_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp(assignvariableop_7_rmsprop_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp#assignvariableop_8_rmsprop_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_rmsprop_rhoIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_14AssignVariableOp7assignvariableop_14_rmsprop_embedding_11_embeddings_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_15AssignVariableOp/assignvariableop_15_rmsprop_dense_21_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp-assignvariableop_16_rmsprop_dense_21_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_17AssignVariableOp/assignvariableop_17_rmsprop_dense_22_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp-assignvariableop_18_rmsprop_dense_22_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ρ
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: ή
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Υ
ς
$__inference_signature_wrapper_519649
embedding_11_input
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCallμ
StatefulPartitionedCallStatefulPartitionedCallembedding_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_519386o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_11_input
Σ

#__inference_internal_grad_fn_519937
result_grads_0
result_grads_1
mul_dense_21_beta
mul_dense_21_biasadd
identityv
mulMulmul_dense_21_betamul_dense_21_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????g
mul_1Mulmul_dense_21_betamul_dense_21_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????

s
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519783

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ε

)__inference_dense_22_layer_call_fn_519823

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_519459o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή*

!__inference__wrapped_model_519386
embedding_11_inputF
2sequential_11_embedding_11_embedding_lookup_519358:
θΰ G
5sequential_11_dense_21_matmul_readvariableop_resource: D
6sequential_11_dense_21_biasadd_readvariableop_resource:G
5sequential_11_dense_22_matmul_readvariableop_resource:D
6sequential_11_dense_22_biasadd_readvariableop_resource:
identity’-sequential_11/dense_21/BiasAdd/ReadVariableOp’,sequential_11/dense_21/MatMul/ReadVariableOp’-sequential_11/dense_22/BiasAdd/ReadVariableOp’,sequential_11/dense_22/MatMul/ReadVariableOp’+sequential_11/embedding_11/embedding_lookup
sequential_11/embedding_11/CastCastembedding_11_input*

DstT0*

SrcT0*0
_output_shapes
:??????????????????°
+sequential_11/embedding_11/embedding_lookupResourceGather2sequential_11_embedding_11_embedding_lookup_519358#sequential_11/embedding_11/Cast:y:0*
Tindices0*E
_class;
97loc:@sequential_11/embedding_11/embedding_lookup/519358*4
_output_shapes"
 :?????????????????? *
dtype0ό
4sequential_11/embedding_11/embedding_lookup/IdentityIdentity4sequential_11/embedding_11/embedding_lookup:output:0*
T0*E
_class;
97loc:@sequential_11/embedding_11/embedding_lookup/519358*4
_output_shapes"
 :?????????????????? ΐ
6sequential_11/embedding_11/embedding_lookup/Identity_1Identity=sequential_11/embedding_11/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? 
@sequential_11/global_average_pooling1d_10/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :τ
.sequential_11/global_average_pooling1d_10/MeanMean?sequential_11/embedding_11/embedding_lookup/Identity_1:output:0Isequential_11/global_average_pooling1d_10/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ’
,sequential_11/dense_21/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Θ
sequential_11/dense_21/MatMulMatMul7sequential_11/global_average_pooling1d_10/Mean:output:04sequential_11/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 
-sequential_11/dense_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_11/dense_21/BiasAddBiasAdd'sequential_11/dense_21/MatMul:product:05sequential_11/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
sequential_11/dense_21/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?’
sequential_11/dense_21/mulMul$sequential_11/dense_21/beta:output:0'sequential_11/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????{
sequential_11/dense_21/SigmoidSigmoidsequential_11/dense_21/mul:z:0*
T0*'
_output_shapes
:?????????’
sequential_11/dense_21/mul_1Mul'sequential_11/dense_21/BiasAdd:output:0"sequential_11/dense_21/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
sequential_11/dense_21/IdentityIdentity sequential_11/dense_21/mul_1:z:0*
T0*'
_output_shapes
:?????????ο
 sequential_11/dense_21/IdentityN	IdentityN sequential_11/dense_21/mul_1:z:0'sequential_11/dense_21/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-519372*:
_output_shapes(
&:?????????:?????????’
,sequential_11/dense_22/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ί
sequential_11/dense_22/MatMulMatMul)sequential_11/dense_21/IdentityN:output:04sequential_11/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 
-sequential_11/dense_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_11/dense_22/BiasAddBiasAdd'sequential_11/dense_22/MatMul:product:05sequential_11/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
IdentityIdentity'sequential_11/dense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????²
NoOpNoOp.^sequential_11/dense_21/BiasAdd/ReadVariableOp-^sequential_11/dense_21/MatMul/ReadVariableOp.^sequential_11/dense_22/BiasAdd/ReadVariableOp-^sequential_11/dense_22/MatMul/ReadVariableOp,^sequential_11/embedding_11/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2^
-sequential_11/dense_21/BiasAdd/ReadVariableOp-sequential_11/dense_21/BiasAdd/ReadVariableOp2\
,sequential_11/dense_21/MatMul/ReadVariableOp,sequential_11/dense_21/MatMul/ReadVariableOp2^
-sequential_11/dense_22/BiasAdd/ReadVariableOp-sequential_11/dense_22/BiasAdd/ReadVariableOp2\
,sequential_11/dense_22/MatMul/ReadVariableOp,sequential_11/dense_22/MatMul/ReadVariableOp2Z
+sequential_11/embedding_11/embedding_lookup+sequential_11/embedding_11/embedding_lookup:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_11_input
Ε

)__inference_dense_21_layer_call_fn_519792

inputs
unknown: 
	unknown_0:
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_519443o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
γ
π
.__inference_sequential_11_layer_call_fn_519668

inputs
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_519470o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs

Ά
I__inference_sequential_11_layer_call_and_return_conditional_losses_519470

inputs'
embedding_11_519417:
θΰ !
dense_21_519444: 
dense_21_519446:!
dense_22_519460:
dense_22_519462:
identity’ dense_21/StatefulPartitionedCall’1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp’ dense_22/StatefulPartitionedCall’$embedding_11/StatefulPartitionedCallω
$embedding_11/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_11_519417*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_embedding_11_layer_call_and_return_conditional_losses_519416
+global_average_pooling1d_10/PartitionedCallPartitionedCall-embedding_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519396‘
 dense_21/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_10/PartitionedCall:output:0dense_21_519444dense_21_519446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_519443
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_519460dense_22_519462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_519459
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_519444*
_output_shapes

: *
dtype0
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£< 
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????η
NoOpNoOp!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_22/StatefulPartitionedCall%^embedding_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
£
«
D__inference_dense_21_layer_call_and_return_conditional_losses_519443

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:

identity_1’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????ͺ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-519431*:
_output_shapes(
&:?????????:?????????
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£< 
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:?????????«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
Σ

-__inference_embedding_11_layer_call_fn_519762

inputs
unknown:
θΰ 
identity’StatefulPartitionedCallΰ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_embedding_11_layer_call_and_return_conditional_losses_519416|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
	
ό
.__inference_sequential_11_layer_call_fn_519483
embedding_11_input
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_519470o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_11_input

z
#__inference_internal_grad_fn_519901
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
γ
π
.__inference_sequential_11_layer_call_fn_519683

inputs
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_519550o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
£
«
D__inference_dense_21_layer_call_and_return_conditional_losses_519814

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:

identity_1’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????ͺ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-519802*:
_output_shapes(
&:?????????:?????????
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£< 
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:?????????«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs<
#__inference_internal_grad_fn_519901CustomGradient-519802<
#__inference_internal_grad_fn_519919CustomGradient-519431<
#__inference_internal_grad_fn_519937CustomGradient-519737<
#__inference_internal_grad_fn_519955CustomGradient-519701<
#__inference_internal_grad_fn_519973CustomGradient-519372"΅	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Κ
serving_defaultΆ
Z
embedding_11_inputD
$serving_default_embedding_11_input:0??????????????????<
dense_220
StatefulPartitionedCall:0?????????tensorflow/serving/predict:¦
θ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
΅
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
₯
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
»
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
C
0
!1
"2
)3
*4"
trackable_list_wrapper
C
0
!1
"2
)3
*4"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
Κ
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ν
1trace_0
2trace_1
3trace_2
4trace_32
.__inference_sequential_11_layer_call_fn_519483
.__inference_sequential_11_layer_call_fn_519668
.__inference_sequential_11_layer_call_fn_519683
.__inference_sequential_11_layer_call_fn_519578Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z1trace_0z2trace_1z3trace_2z4trace_3
Ω
5trace_0
6trace_1
7trace_2
8trace_32ξ
I__inference_sequential_11_layer_call_and_return_conditional_losses_519719
I__inference_sequential_11_layer_call_and_return_conditional_losses_519755
I__inference_sequential_11_layer_call_and_return_conditional_losses_519600
I__inference_sequential_11_layer_call_and_return_conditional_losses_519622Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z5trace_0z6trace_1z7trace_2z8trace_3
ΧBΤ
!__inference__wrapped_model_519386embedding_11_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 

9iter
	:decay
;learning_rate
<momentum
=rho	rmsg	!rmsh	"rmsi	)rmsj	*rmsk"
	optimizer
,
>serving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ρ
Dtrace_02Τ
-__inference_embedding_11_layer_call_fn_519762’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zDtrace_0

Etrace_02ο
H__inference_embedding_11_layer_call_and_return_conditional_losses_519772’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zEtrace_0
+:)
θΰ 2embedding_11/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object

Ktrace_02π
<__inference_global_average_pooling1d_10_layer_call_fn_519777―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zKtrace_0
¨
Ltrace_02
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519783―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zLtrace_0
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
ν
Rtrace_02Π
)__inference_dense_21_layer_call_fn_519792’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zRtrace_0

Strace_02λ
D__inference_dense_21_layer_call_and_return_conditional_losses_519814’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zStrace_0
!: 2dense_21/kernel
:2dense_21/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ν
Ytrace_02Π
)__inference_dense_22_layer_call_fn_519823’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zYtrace_0

Ztrace_02λ
D__inference_dense_22_layer_call_and_return_conditional_losses_519833’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zZtrace_0
!:2dense_22/kernel
:2dense_22/bias
Ν
[trace_02°
__inference_loss_fn_0_519842
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *’ z[trace_0
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_11_layer_call_fn_519483embedding_11_input"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
?Bό
.__inference_sequential_11_layer_call_fn_519668inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
?Bό
.__inference_sequential_11_layer_call_fn_519683inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
.__inference_sequential_11_layer_call_fn_519578embedding_11_input"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
I__inference_sequential_11_layer_call_and_return_conditional_losses_519719inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
I__inference_sequential_11_layer_call_and_return_conditional_losses_519755inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¦B£
I__inference_sequential_11_layer_call_and_return_conditional_losses_519600embedding_11_input"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¦B£
I__inference_sequential_11_layer_call_and_return_conditional_losses_519622embedding_11_input"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
ΦBΣ
$__inference_signature_wrapper_519649embedding_11_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
αBή
-__inference_embedding_11_layer_call_fn_519762inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
όBω
H__inference_embedding_11_layer_call_and_return_conditional_losses_519772inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ύBϊ
<__inference_global_average_pooling1d_10_layer_call_fn_519777inputs"―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519783inputs"―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_dense_21_layer_call_fn_519792inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_dense_21_layer_call_and_return_conditional_losses_519814inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_dense_22_layer_call_fn_519823inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_dense_22_layer_call_and_return_conditional_losses_519833inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
³B°
__inference_loss_fn_0_519842"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *’ 
N
^	variables
_	keras_api
	`total
	acount"
_tf_keras_metric
^
b	variables
c	keras_api
	dtotal
	ecount
f
_fn_kwargs"
_tf_keras_metric
.
`0
a1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
:  (2total
:  (2count
.
d0
e1"
trackable_list_wrapper
-
b	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
5:3
θΰ 2#RMSprop/embedding_11/embeddings/rms
+:) 2RMSprop/dense_21/kernel/rms
%:#2RMSprop/dense_21/bias/rms
+:)2RMSprop/dense_22/kernel/rms
%:#2RMSprop/dense_22/bias/rms
PbN
beta:0D__inference_dense_21_layer_call_and_return_conditional_losses_519814
SbQ
	BiasAdd:0D__inference_dense_21_layer_call_and_return_conditional_losses_519814
PbN
beta:0D__inference_dense_21_layer_call_and_return_conditional_losses_519443
SbQ
	BiasAdd:0D__inference_dense_21_layer_call_and_return_conditional_losses_519443
^b\
dense_21/beta:0I__inference_sequential_11_layer_call_and_return_conditional_losses_519755
ab_
dense_21/BiasAdd:0I__inference_sequential_11_layer_call_and_return_conditional_losses_519755
^b\
dense_21/beta:0I__inference_sequential_11_layer_call_and_return_conditional_losses_519719
ab_
dense_21/BiasAdd:0I__inference_sequential_11_layer_call_and_return_conditional_losses_519719
DbB
sequential_11/dense_21/beta:0!__inference__wrapped_model_519386
GbE
 sequential_11/dense_21/BiasAdd:0!__inference__wrapped_model_519386¨
!__inference__wrapped_model_519386!")*D’A
:’7
52
embedding_11_input??????????????????
ͺ "3ͺ0
.
dense_22"
dense_22?????????€
D__inference_dense_21_layer_call_and_return_conditional_losses_519814\!"/’,
%’"
 
inputs????????? 
ͺ "%’"

0?????????
 |
)__inference_dense_21_layer_call_fn_519792O!"/’,
%’"
 
inputs????????? 
ͺ "?????????€
D__inference_dense_22_layer_call_and_return_conditional_losses_519833\)*/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 |
)__inference_dense_22_layer_call_fn_519823O)*/’,
%’"
 
inputs?????????
ͺ "?????????½
H__inference_embedding_11_layer_call_and_return_conditional_losses_519772q8’5
.’+
)&
inputs??????????????????
ͺ "2’/
(%
0?????????????????? 
 
-__inference_embedding_11_layer_call_fn_519762d8’5
.’+
)&
inputs??????????????????
ͺ "%"?????????????????? Φ
W__inference_global_average_pooling1d_10_layer_call_and_return_conditional_losses_519783{I’F
?’<
63
inputs'???????????????????????????

 
ͺ ".’+
$!
0??????????????????
 ?
<__inference_global_average_pooling1d_10_layer_call_fn_519777nI’F
?’<
63
inputs'???????????????????????????

 
ͺ "!??????????????????Ή
#__inference_internal_grad_fn_519901lme’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_519919noe’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_519937pqe’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_519955rse’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_519973tue’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????;
__inference_loss_fn_0_519842!’

’ 
ͺ " Ι
I__inference_sequential_11_layer_call_and_return_conditional_losses_519600|!")*L’I
B’?
52
embedding_11_input??????????????????
p 

 
ͺ "%’"

0?????????
 Ι
I__inference_sequential_11_layer_call_and_return_conditional_losses_519622|!")*L’I
B’?
52
embedding_11_input??????????????????
p

 
ͺ "%’"

0?????????
 ½
I__inference_sequential_11_layer_call_and_return_conditional_losses_519719p!")*@’=
6’3
)&
inputs??????????????????
p 

 
ͺ "%’"

0?????????
 ½
I__inference_sequential_11_layer_call_and_return_conditional_losses_519755p!")*@’=
6’3
)&
inputs??????????????????
p

 
ͺ "%’"

0?????????
 ‘
.__inference_sequential_11_layer_call_fn_519483o!")*L’I
B’?
52
embedding_11_input??????????????????
p 

 
ͺ "?????????‘
.__inference_sequential_11_layer_call_fn_519578o!")*L’I
B’?
52
embedding_11_input??????????????????
p

 
ͺ "?????????
.__inference_sequential_11_layer_call_fn_519668c!")*@’=
6’3
)&
inputs??????????????????
p 

 
ͺ "?????????
.__inference_sequential_11_layer_call_fn_519683c!")*@’=
6’3
)&
inputs??????????????????
p

 
ͺ "?????????Α
$__inference_signature_wrapper_519649!")*Z’W
’ 
PͺM
K
embedding_11_input52
embedding_11_input??????????????????"3ͺ0
.
dense_22"
dense_22?????????