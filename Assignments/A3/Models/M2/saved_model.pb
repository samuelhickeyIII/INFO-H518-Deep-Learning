Κό
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
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ηΥ
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

: *
dtype0

Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
θΰ *.
shared_nameAdam/embedding_1/embeddings/v

1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v* 
_output_shapes
:
θΰ *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

: *
dtype0

Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
θΰ *.
shared_nameAdam/embedding_1/embeddings/m

1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m* 
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0

embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
θΰ *'
shared_nameembedding_1/embeddings

*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings* 
_output_shapes
:
θΰ *
dtype0

!serving_default_embedding_1_inputPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????

StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_1_inputembedding_1/embeddingsdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
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
$__inference_signature_wrapper_101719

NoOpNoOp
­0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*θ/
valueή/BΫ/ BΤ/
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

9iter

:beta_1

;beta_2
	<decay
=learning_ratemg!mh"mi)mj*mkvl!vm"vn)vo*vp*
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
jd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
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
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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

VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
½	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
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
__inference__traced_save_102097
Ψ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/embedding_1/embeddings/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/embedding_1/embeddings/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*$
Tin
2*
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
"__inference__traced_restore_102179Οδ

·
H__inference_sequential_1_layer_call_and_return_conditional_losses_101692
embedding_1_input&
embedding_1_101673:
θΰ  
dense_1_101677: 
dense_1_101679: 
dense_2_101682:
dense_2_101684:
identity’dense_1/StatefulPartitionedCall’0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp’dense_2/StatefulPartitionedCall’#embedding_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_101673*
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
GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_101486
(global_average_pooling1d/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101466
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1_101677dense_1_101679*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101513
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_101682dense_2_101684*
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
GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101529
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_101677*
_output_shapes

: *
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£<
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????γ
NoOpNoOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
ε	
¦
G__inference_embedding_1_layer_call_and_return_conditional_losses_101842

inputs+
embedding_lookup_101836:
θΰ 
identity’embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????Δ
embedding_lookupResourceGatherembedding_lookup_101836Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/101836*4
_output_shapes"
 :?????????????????? *
dtype0«
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/101836*4
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
α
ο
-__inference_sequential_1_layer_call_fn_101738

inputs
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
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
GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_101540o
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
a

"__inference__traced_restore_102179
file_prefix;
'assignvariableop_embedding_1_embeddings:
θΰ 3
!assignvariableop_1_dense_1_kernel: -
assignvariableop_2_dense_1_bias:3
!assignvariableop_3_dense_2_kernel:-
assignvariableop_4_dense_2_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: E
1assignvariableop_14_adam_embedding_1_embeddings_m:
θΰ ;
)assignvariableop_15_adam_dense_1_kernel_m: 5
'assignvariableop_16_adam_dense_1_bias_m:;
)assignvariableop_17_adam_dense_2_kernel_m:5
'assignvariableop_18_adam_dense_2_bias_m:E
1assignvariableop_19_adam_embedding_1_embeddings_v:
θΰ ;
)assignvariableop_20_adam_dense_1_kernel_v: 5
'assignvariableop_21_adam_dense_1_bias_v:;
)assignvariableop_22_adam_dense_2_kernel_v:5
'assignvariableop_23_adam_dense_2_bias_v:
identity_25’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9ͺ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Π
valueΖBΓB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH’
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_2_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_2_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
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
:’
AssignVariableOp_14AssignVariableOp1assignvariableop_14_adam_embedding_1_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:’
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_embedding_1_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_1_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_1_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_2_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_2_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ί
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: Μ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
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

€
#__inference_internal_grad_fn_102053
result_grads_0
result_grads_1!
mul_sequential_1_dense_1_beta$
 mul_sequential_1_dense_1_biasadd
identity
mulMulmul_sequential_1_dense_1_beta mul_sequential_1_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????
mul_1Mulmul_sequential_1_dense_1_beta mul_sequential_1_dense_1_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????

©
C__inference_dense_1_layer_call_and_return_conditional_losses_101513

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:

identity_1’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????ͺ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101501*:
_output_shapes(
&:?????????:?????????
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£<
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:?????????ͺ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
Ν

#__inference_internal_grad_fn_102035
result_grads_0
result_grads_1
mul_dense_1_beta
mul_dense_1_biasadd
identityt
mulMulmul_dense_1_betamul_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????e
mul_1Mulmul_dense_1_betamul_dense_1_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
Ζ	
τ
C__inference_dense_2_layer_call_and_return_conditional_losses_101903

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs

p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101466

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
7
ϋ	
__inference__traced_save_102097
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
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
: §
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Π
valueΖBΓB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B χ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	
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

identity_1Identity_1:output:0*―
_input_shapes
: :
θΰ : :::: : : : : : : : : :
θΰ : ::::
θΰ : :::: 2(
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

: : 

_output_shapes
::$ 

_output_shapes

:: 
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

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::&"
 
_output_shapes
:
θΰ :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
ε
¬
H__inference_sequential_1_layer_call_and_return_conditional_losses_101620

inputs&
embedding_1_101601:
θΰ  
dense_1_101605: 
dense_1_101607: 
dense_2_101610:
dense_2_101612:
identity’dense_1/StatefulPartitionedCall’0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp’dense_2/StatefulPartitionedCall’#embedding_1/StatefulPartitionedCallφ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_101601*
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
GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_101486
(global_average_pooling1d/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101466
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1_101605dense_1_101607*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101513
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_101610dense_2_101612*
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
GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101529
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_101605*
_output_shapes

: *
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£<
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????γ
NoOpNoOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
ε
¬
H__inference_sequential_1_layer_call_and_return_conditional_losses_101540

inputs&
embedding_1_101487:
θΰ  
dense_1_101514: 
dense_1_101516: 
dense_2_101530:
dense_2_101532:
identity’dense_1/StatefulPartitionedCall’0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp’dense_2/StatefulPartitionedCall’#embedding_1/StatefulPartitionedCallφ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_101487*
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
GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_101486
(global_average_pooling1d/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101466
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1_101514dense_1_101516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101513
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_101530dense_2_101532*
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
GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101529
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_101514*
_output_shapes

: *
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£<
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????γ
NoOpNoOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
	
ϊ
-__inference_sequential_1_layer_call_fn_101553
embedding_1_input
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_101540o
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
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input

z
#__inference_internal_grad_fn_101999
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
ώ
U
9__inference_global_average_pooling1d_layer_call_fn_101847

inputs
identityΛ
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101466i
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

·
H__inference_sequential_1_layer_call_and_return_conditional_losses_101670
embedding_1_input&
embedding_1_101651:
θΰ  
dense_1_101655: 
dense_1_101657: 
dense_2_101660:
dense_2_101662:
identity’dense_1/StatefulPartitionedCall’0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp’dense_2/StatefulPartitionedCall’#embedding_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_101651*
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
GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_101486
(global_average_pooling1d/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101466
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1_101655dense_1_101657*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101513
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_101660dense_2_101662*
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
GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101529
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_101655*
_output_shapes

: *
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£<
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????γ
NoOpNoOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
Ρ

,__inference_embedding_1_layer_call_fn_101832

inputs
unknown:
θΰ 
identity’StatefulPartitionedCallί
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
GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_101486|
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
Γ

(__inference_dense_1_layer_call_fn_101862

inputs
unknown: 
	unknown_0:
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101513o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
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
	
ϊ
-__inference_sequential_1_layer_call_fn_101648
embedding_1_input
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_101620o
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
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
Ν

#__inference_internal_grad_fn_102017
result_grads_0
result_grads_1
mul_dense_1_beta
mul_dense_1_biasadd
identityt
mulMulmul_dense_1_betamul_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????e
mul_1Mulmul_dense_1_betamul_dense_1_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
Π'
Φ
H__inference_sequential_1_layer_call_and_return_conditional_losses_101789

inputs7
#embedding_1_embedding_lookup_101757:
θΰ 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp’dense_2/BiasAdd/ReadVariableOp’dense_2/MatMul/ReadVariableOp’embedding_1/embedding_lookupj
embedding_1/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????τ
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_101757embedding_1/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/101757*4
_output_shapes"
 :?????????????????? *
dtype0Ο
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/101757*4
_output_shapes"
 :?????????????????? ’
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Γ
global_average_pooling1d/MeanMean0embedding_1/embedding_lookup/Identity_1:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_1/MatMulMatMul&global_average_pooling1d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Q
dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
dense_1/mulMuldense_1/beta:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????]
dense_1/SigmoidSigmoiddense_1/mul:z:0*
T0*'
_output_shapes
:?????????u
dense_1/mul_1Muldense_1/BiasAdd:output:0dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????a
dense_1/IdentityIdentitydense_1/mul_1:z:0*
T0*'
_output_shapes
:?????????Β
dense_1/IdentityN	IdentityNdense_1/mul_1:z:0dense_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101771*:
_output_shapes(
&:?????????:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMuldense_1/IdentityN:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£<
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
	
―
__inference_loss_fn_0_101912K
9dense_1_kernel_regularizer_l2loss_readvariableop_resource: 
identity’0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpͺ
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

: *
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£<
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp

©
C__inference_dense_1_layer_call_and_return_conditional_losses_101884

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:

identity_1’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????ͺ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101872*:
_output_shapes(
&:?????????:?????????
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£<
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:?????????ͺ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
Ζ	
τ
C__inference_dense_2_layer_call_and_return_conditional_losses_101529

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Π'
Φ
H__inference_sequential_1_layer_call_and_return_conditional_losses_101825

inputs7
#embedding_1_embedding_lookup_101793:
θΰ 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp’dense_2/BiasAdd/ReadVariableOp’dense_2/MatMul/ReadVariableOp’embedding_1/embedding_lookupj
embedding_1/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????τ
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_101793embedding_1/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/101793*4
_output_shapes"
 :?????????????????? *
dtype0Ο
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/101793*4
_output_shapes"
 :?????????????????? ’
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Γ
global_average_pooling1d/MeanMean0embedding_1/embedding_lookup/Identity_1:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_1/MatMulMatMul&global_average_pooling1d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Q
dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
dense_1/mulMuldense_1/beta:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????]
dense_1/SigmoidSigmoiddense_1/mul:z:0*
T0*'
_output_shapes
:?????????u
dense_1/mul_1Muldense_1/BiasAdd:output:0dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????a
dense_1/IdentityIdentitydense_1/mul_1:z:0*
T0*'
_output_shapes
:?????????Β
dense_1/IdentityN	IdentityNdense_1/mul_1:z:0dense_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101807*:
_output_shapes(
&:?????????:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMuldense_1/IdentityN:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ£<
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs

z
#__inference_internal_grad_fn_101981
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
)

!__inference__wrapped_model_101456
embedding_1_inputD
0sequential_1_embedding_1_embedding_lookup_101428:
θΰ E
3sequential_1_dense_1_matmul_readvariableop_resource: B
4sequential_1_dense_1_biasadd_readvariableop_resource:E
3sequential_1_dense_2_matmul_readvariableop_resource:B
4sequential_1_dense_2_biasadd_readvariableop_resource:
identity’+sequential_1/dense_1/BiasAdd/ReadVariableOp’*sequential_1/dense_1/MatMul/ReadVariableOp’+sequential_1/dense_2/BiasAdd/ReadVariableOp’*sequential_1/dense_2/MatMul/ReadVariableOp’)sequential_1/embedding_1/embedding_lookup
sequential_1/embedding_1/CastCastembedding_1_input*

DstT0*

SrcT0*0
_output_shapes
:??????????????????¨
)sequential_1/embedding_1/embedding_lookupResourceGather0sequential_1_embedding_1_embedding_lookup_101428!sequential_1/embedding_1/Cast:y:0*
Tindices0*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/101428*4
_output_shapes"
 :?????????????????? *
dtype0φ
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/101428*4
_output_shapes"
 :?????????????????? Ό
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? ~
<sequential_1/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :κ
*sequential_1/global_average_pooling1d/MeanMean=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0Esequential_1/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ΐ
sequential_1/dense_1/MatMulMatMul3sequential_1/global_average_pooling1d/Mean:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0΅
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
sequential_1/dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
sequential_1/dense_1/mulMul"sequential_1/dense_1/beta:output:0%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
sequential_1/dense_1/SigmoidSigmoidsequential_1/dense_1/mul:z:0*
T0*'
_output_shapes
:?????????
sequential_1/dense_1/mul_1Mul%sequential_1/dense_1/BiasAdd:output:0 sequential_1/dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????{
sequential_1/dense_1/IdentityIdentitysequential_1/dense_1/mul_1:z:0*
T0*'
_output_shapes
:?????????ι
sequential_1/dense_1/IdentityN	IdentityNsequential_1/dense_1/mul_1:z:0%sequential_1/dense_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101442*:
_output_shapes(
&:?????????:?????????
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0΄
sequential_1/dense_2/MatMulMatMul'sequential_1/dense_1/IdentityN:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0΅
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_1/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????¨
NoOpNoOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp*^sequential_1/embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
Γ

(__inference_dense_2_layer_call_fn_101893

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallΫ
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
GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101529o
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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
ρ
$__inference_signature_wrapper_101719
embedding_1_input
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCallλ
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
!__inference__wrapped_model_101456o
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
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
α
ο
-__inference_sequential_1_layer_call_fn_101753

inputs
unknown:
θΰ 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
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
GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_101620o
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
ε	
¦
G__inference_embedding_1_layer_call_and_return_conditional_losses_101486

inputs+
embedding_lookup_101480:
θΰ 
identity’embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????Δ
embedding_lookupResourceGatherembedding_lookup_101480Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/101480*4
_output_shapes"
 :?????????????????? *
dtype0«
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/101480*4
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

p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101853

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
 
_user_specified_nameinputs<
#__inference_internal_grad_fn_101981CustomGradient-101872<
#__inference_internal_grad_fn_101999CustomGradient-101501<
#__inference_internal_grad_fn_102017CustomGradient-101807<
#__inference_internal_grad_fn_102035CustomGradient-101771<
#__inference_internal_grad_fn_102053CustomGradient-101442"΅	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Η
serving_default³
X
embedding_1_inputC
#serving_default_embedding_1_input:0??????????????????;
dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ό
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
ι
1trace_0
2trace_1
3trace_2
4trace_32ώ
-__inference_sequential_1_layer_call_fn_101553
-__inference_sequential_1_layer_call_fn_101738
-__inference_sequential_1_layer_call_fn_101753
-__inference_sequential_1_layer_call_fn_101648Ώ
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
Υ
5trace_0
6trace_1
7trace_2
8trace_32κ
H__inference_sequential_1_layer_call_and_return_conditional_losses_101789
H__inference_sequential_1_layer_call_and_return_conditional_losses_101825
H__inference_sequential_1_layer_call_and_return_conditional_losses_101670
H__inference_sequential_1_layer_call_and_return_conditional_losses_101692Ώ
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
ΦBΣ
!__inference__wrapped_model_101456embedding_1_input"
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
­
9iter

:beta_1

;beta_2
	<decay
=learning_ratemg!mh"mi)mj*mkvl!vm"vn)vo*vp"
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
π
Dtrace_02Σ
,__inference_embedding_1_layer_call_fn_101832’
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

Etrace_02ξ
G__inference_embedding_1_layer_call_and_return_conditional_losses_101842’
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
*:(
θΰ 2embedding_1/embeddings
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

Ktrace_02ν
9__inference_global_average_pooling1d_layer_call_fn_101847―
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
₯
Ltrace_02
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101853―
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
μ
Rtrace_02Ο
(__inference_dense_1_layer_call_fn_101862’
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

Strace_02κ
C__inference_dense_1_layer_call_and_return_conditional_losses_101884’
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
 : 2dense_1/kernel
:2dense_1/bias
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
μ
Ytrace_02Ο
(__inference_dense_2_layer_call_fn_101893’
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

Ztrace_02κ
C__inference_dense_2_layer_call_and_return_conditional_losses_101903’
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
 :2dense_2/kernel
:2dense_2/bias
Ν
[trace_02°
__inference_loss_fn_0_101912
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
B
-__inference_sequential_1_layer_call_fn_101553embedding_1_input"Ώ
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
ώBϋ
-__inference_sequential_1_layer_call_fn_101738inputs"Ώ
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
ώBϋ
-__inference_sequential_1_layer_call_fn_101753inputs"Ώ
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
B
-__inference_sequential_1_layer_call_fn_101648embedding_1_input"Ώ
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
B
H__inference_sequential_1_layer_call_and_return_conditional_losses_101789inputs"Ώ
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
B
H__inference_sequential_1_layer_call_and_return_conditional_losses_101825inputs"Ώ
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
€B‘
H__inference_sequential_1_layer_call_and_return_conditional_losses_101670embedding_1_input"Ώ
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
€B‘
H__inference_sequential_1_layer_call_and_return_conditional_losses_101692embedding_1_input"Ώ
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ΥB?
$__inference_signature_wrapper_101719embedding_1_input"
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
ΰBέ
,__inference_embedding_1_layer_call_fn_101832inputs"’
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
ϋBψ
G__inference_embedding_1_layer_call_and_return_conditional_losses_101842inputs"’
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
ϊBχ
9__inference_global_average_pooling1d_layer_call_fn_101847inputs"―
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
B
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101853inputs"―
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
άBΩ
(__inference_dense_1_layer_call_fn_101862inputs"’
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
χBτ
C__inference_dense_1_layer_call_and_return_conditional_losses_101884inputs"’
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
άBΩ
(__inference_dense_2_layer_call_fn_101893inputs"’
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
χBτ
C__inference_dense_2_layer_call_and_return_conditional_losses_101903inputs"’
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
__inference_loss_fn_0_101912"
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
/:-
θΰ 2Adam/embedding_1/embeddings/m
%:# 2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
/:-
θΰ 2Adam/embedding_1/embeddings/v
%:# 2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
ObM
beta:0C__inference_dense_1_layer_call_and_return_conditional_losses_101884
RbP
	BiasAdd:0C__inference_dense_1_layer_call_and_return_conditional_losses_101884
ObM
beta:0C__inference_dense_1_layer_call_and_return_conditional_losses_101513
RbP
	BiasAdd:0C__inference_dense_1_layer_call_and_return_conditional_losses_101513
\bZ
dense_1/beta:0H__inference_sequential_1_layer_call_and_return_conditional_losses_101825
_b]
dense_1/BiasAdd:0H__inference_sequential_1_layer_call_and_return_conditional_losses_101825
\bZ
dense_1/beta:0H__inference_sequential_1_layer_call_and_return_conditional_losses_101789
_b]
dense_1/BiasAdd:0H__inference_sequential_1_layer_call_and_return_conditional_losses_101789
Bb@
sequential_1/dense_1/beta:0!__inference__wrapped_model_101456
EbC
sequential_1/dense_1/BiasAdd:0!__inference__wrapped_model_101456€
!__inference__wrapped_model_101456!")*C’@
9’6
41
embedding_1_input??????????????????
ͺ "1ͺ.
,
dense_2!
dense_2?????????£
C__inference_dense_1_layer_call_and_return_conditional_losses_101884\!"/’,
%’"
 
inputs????????? 
ͺ "%’"

0?????????
 {
(__inference_dense_1_layer_call_fn_101862O!"/’,
%’"
 
inputs????????? 
ͺ "?????????£
C__inference_dense_2_layer_call_and_return_conditional_losses_101903\)*/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 {
(__inference_dense_2_layer_call_fn_101893O)*/’,
%’"
 
inputs?????????
ͺ "?????????Ό
G__inference_embedding_1_layer_call_and_return_conditional_losses_101842q8’5
.’+
)&
inputs??????????????????
ͺ "2’/
(%
0?????????????????? 
 
,__inference_embedding_1_layer_call_fn_101832d8’5
.’+
)&
inputs??????????????????
ͺ "%"?????????????????? Σ
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_101853{I’F
?’<
63
inputs'???????????????????????????

 
ͺ ".’+
$!
0??????????????????
 «
9__inference_global_average_pooling1d_layer_call_fn_101847nI’F
?’<
63
inputs'???????????????????????????

 
ͺ "!??????????????????Ή
#__inference_internal_grad_fn_101981qre’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_101999ste’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_102017uve’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_102035wxe’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_102053yze’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????;
__inference_loss_fn_0_101912!’

’ 
ͺ " Η
H__inference_sequential_1_layer_call_and_return_conditional_losses_101670{!")*K’H
A’>
41
embedding_1_input??????????????????
p 

 
ͺ "%’"

0?????????
 Η
H__inference_sequential_1_layer_call_and_return_conditional_losses_101692{!")*K’H
A’>
41
embedding_1_input??????????????????
p

 
ͺ "%’"

0?????????
 Ό
H__inference_sequential_1_layer_call_and_return_conditional_losses_101789p!")*@’=
6’3
)&
inputs??????????????????
p 

 
ͺ "%’"

0?????????
 Ό
H__inference_sequential_1_layer_call_and_return_conditional_losses_101825p!")*@’=
6’3
)&
inputs??????????????????
p

 
ͺ "%’"

0?????????
 
-__inference_sequential_1_layer_call_fn_101553n!")*K’H
A’>
41
embedding_1_input??????????????????
p 

 
ͺ "?????????
-__inference_sequential_1_layer_call_fn_101648n!")*K’H
A’>
41
embedding_1_input??????????????????
p

 
ͺ "?????????
-__inference_sequential_1_layer_call_fn_101738c!")*@’=
6’3
)&
inputs??????????????????
p 

 
ͺ "?????????
-__inference_sequential_1_layer_call_fn_101753c!")*@’=
6’3
)&
inputs??????????????????
p

 
ͺ "?????????½
$__inference_signature_wrapper_101719!")*X’U
’ 
NͺK
I
embedding_1_input41
embedding_1_input??????????????????"1ͺ.
,
dense_2!
dense_2?????????