Éà0
¬ü
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
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
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8 .
â
EAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*V
shared_nameGEAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/v
Û
YAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/v/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/v*
_output_shapes
:@*
dtype0
þ
QAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*b
shared_nameSQAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/v
÷
eAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpQAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
ê
GAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*X
shared_nameIGAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/v
ã
[Adam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/v*
_output_shapes

:4@*
dtype0
à
DAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*U
shared_nameFDAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/v
Ù
XAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/v/Read/ReadVariableOpReadVariableOpDAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/v*
_output_shapes
:@*
dtype0
ü
PAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*a
shared_nameRPAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/v
õ
dAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpPAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
è
FAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*W
shared_nameHFAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/v
á
ZAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/v*
_output_shapes

:4@*
dtype0

Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:*
dtype0

Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_15/kernel/v

*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes
:	*
dtype0
â
EAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*V
shared_nameGEAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/m
Û
YAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/m/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/m*
_output_shapes
:@*
dtype0
þ
QAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*b
shared_nameSQAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/m
÷
eAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpQAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
ê
GAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*X
shared_nameIGAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/m
ã
[Adam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/m*
_output_shapes

:4@*
dtype0
à
DAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*U
shared_nameFDAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/m
Ù
XAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/m/Read/ReadVariableOpReadVariableOpDAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/m*
_output_shapes
:@*
dtype0
ü
PAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*a
shared_nameRPAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/m
õ
dAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpPAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
è
FAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*W
shared_nameHFAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/m
á
ZAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/m*
_output_shapes

:4@*
dtype0

Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:*
dtype0

Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_15/kernel/m

*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes
:	*
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
Ô
>bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*O
shared_name@>bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias
Í
Rbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/Read/ReadVariableOpReadVariableOp>bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias*
_output_shapes
:@*
dtype0
ð
Jbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*[
shared_nameLJbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel
é
^bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/Read/ReadVariableOpReadVariableOpJbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel*
_output_shapes

:@@*
dtype0
Ü
@bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*Q
shared_nameB@bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel
Õ
Tbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/Read/ReadVariableOpReadVariableOp@bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel*
_output_shapes

:4@*
dtype0
Ò
=bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*N
shared_name?=bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias
Ë
Qbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/Read/ReadVariableOpReadVariableOp=bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias*
_output_shapes
:@*
dtype0
î
Ibidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*Z
shared_nameKIbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel
ç
]bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/Read/ReadVariableOpReadVariableOpIbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel*
_output_shapes

:@@*
dtype0
Ú
?bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*P
shared_nameA?bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel
Ó
Sbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/Read/ReadVariableOpReadVariableOp?bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel*
_output_shapes

:4@*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes
:	*
dtype0

&serving_default_bidirectional_15_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ4

StatefulPartitionedCallStatefulPartitionedCall&serving_default_bidirectional_15_input?bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel=bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/biasIbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel@bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel>bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/biasJbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kerneldense_15/kerneldense_15/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_4508931

NoOpNoOp
èL
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*£L
valueLBL BL

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
·
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
forward_layer
backward_layer*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
<
0
1
2
3
 4
!5
6
7*
<
0
1
2
3
 4
!5
6
7*
* 
°
"non_trainable_variables

#layers
$metrics
%layer_regularization_losses
&layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
'trace_0
(trace_1
)trace_2
*trace_3* 
6
+trace_0
,trace_1
-trace_2
.trace_3* 
* 
ä
/iter

0beta_1

1beta_2
	2decay
3learning_ratem m¡m¢m£m¤m¥ m¦!m§v¨v©vªv«v¬v­ v®!v¯*

4serving_default* 
.
0
1
2
3
 4
!5*
.
0
1
2
3
 4
!5*
* 

5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
:trace_0
;trace_1
<trace_2
=trace_3* 
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
ª
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Hcell
I
state_spec*
ª
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Pcell
Q
state_spec*

0
1*

0
1*
* 

Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Wtrace_0* 

Xtrace_0* 
_Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_15/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEIbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE@bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

Y0
Z1*
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

0
1*
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

0
1
2*

0
1
2*
* 


[states
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
6
atrace_0
btrace_1
ctrace_2
dtrace_3* 
6
etrace_0
ftrace_1
gtrace_2
htrace_3* 
Ó
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
o_random_generator

kernel
recurrent_kernel
bias*
* 

0
 1
!2*

0
 1
!2*
* 


pstates
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
6
vtrace_0
wtrace_1
xtrace_2
ytrace_3* 
6
ztrace_0
{trace_1
|trace_2
}trace_3* 
Ø
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

kernel
 recurrent_kernel
!bias*
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
* 
* 

H0*
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

0
1
2*

0
1
2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 

P0*
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

0
 1
!2*

0
 1
!2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
£
VARIABLE_VALUEFAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
­¦
VARIABLE_VALUEPAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¡
VARIABLE_VALUEDAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
£
VARIABLE_VALUEFAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
­¦
VARIABLE_VALUEPAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¡
VARIABLE_VALUEDAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ô
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpSbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/Read/ReadVariableOp]bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/Read/ReadVariableOpQbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/Read/ReadVariableOpTbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/Read/ReadVariableOp^bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/Read/ReadVariableOpRbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOpZAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/m/Read/ReadVariableOpdAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/m/Read/ReadVariableOpXAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/m/Read/ReadVariableOp[Adam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/m/Read/ReadVariableOpeAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/m/Read/ReadVariableOpYAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOpZAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/v/Read/ReadVariableOpdAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/v/Read/ReadVariableOpXAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/v/Read/ReadVariableOp[Adam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/v/Read/ReadVariableOpeAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/v/Read/ReadVariableOpYAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_4511617
»
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_15/kerneldense_15/bias?bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernelIbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel=bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias@bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernelJbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel>bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_15/kernel/mAdam/dense_15/bias/mFAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/mPAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/mDAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/mGAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/mQAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/mEAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/mAdam/dense_15/kernel/vAdam/dense_15/bias/vFAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/vPAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/vDAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/vGAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/vQAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/vEAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/v*-
Tin&
$2"*
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_4511726ÌÍ,
ÛA
É
'forward_simple_rnn_8_while_body_4508281F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2E
Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0
}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_8_while_identity)
%forward_simple_rnn_8_while_identity_1)
%forward_simple_rnn_8_while_identity_2)
%forward_simple_rnn_8_while_identity_3)
%forward_simple_rnn_8_while_identity_4C
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp¢Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
Lforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
>forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_8_while_placeholderUforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ò
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_8/while/simple_rnn_cell_25/MatMulMatMulEforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAddBiasAdd>forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul:product:0Lforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1MatMul(forward_simple_rnn_8_while_placeholder_2Mforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_8/while/simple_rnn_cell_25/addAddV2>forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd:output:0@forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_8/while/simple_rnn_cell_25/TanhTanh5forward_simple_rnn_8/while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_8_while_placeholder_1Nforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_8/while/addAddV2&forward_simple_rnn_8_while_placeholder)forward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_8/while/add_1AddV2Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counter+forward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_8/while/IdentityIdentity$forward_simple_rnn_8/while/add_1:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_8/while/Identity_1IdentityHforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_8/while/Identity_2Identity"forward_simple_rnn_8/while/add:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_8/while/Identity_3IdentityOforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_8/while/Identity_4Identity6forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0 ^forward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_8/while/NoOpNoOpE^forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpD^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpF^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0"S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0"W
%forward_simple_rnn_8_while_identity_1.forward_simple_rnn_8/while/Identity_1:output:0"W
%forward_simple_rnn_8_while_identity_2.forward_simple_rnn_8/while/Identity_2:output:0"W
%forward_simple_rnn_8_while_identity_3.forward_simple_rnn_8/while/Identity_3:output:0"W
%forward_simple_rnn_8_while_identity_4.forward_simple_rnn_8/while/Identity_4:output:0" 
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resourceOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resourcePforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resourceNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpDforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpCforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp2
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpEforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
	

2__inference_bidirectional_15_layer_call_fn_4509478

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508458p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
¹
Á
7__inference_backward_simple_rnn_8_layer_call_fn_4510912

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4507903o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
'forward_simple_rnn_8_while_cond_4508280F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2H
Dforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4508280___redundant_placeholder0_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4508280___redundant_placeholder1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4508280___redundant_placeholder2_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4508280___redundant_placeholder3'
#forward_simple_rnn_8_while_identity
¶
forward_simple_rnn_8/while/LessLess&forward_simple_rnn_8_while_placeholderDforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_8/while/IdentityIdentity#forward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:

¾
'forward_simple_rnn_8_while_cond_4509757F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2H
Dforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509757___redundant_placeholder0_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509757___redundant_placeholder1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509757___redundant_placeholder2_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509757___redundant_placeholder3'
#forward_simple_rnn_8_while_identity
¶
forward_simple_rnn_8/while/LessLess&forward_simple_rnn_8_while_placeholderDforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_8/while/IdentityIdentity#forward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ÔB
è
(backward_simple_rnn_8_while_body_4510306H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2G
Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0
backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_8_while_identity*
&backward_simple_rnn_8_while_identity_1*
&backward_simple_rnn_8_while_identity_2*
&backward_simple_rnn_8_while_identity_3*
&backward_simple_rnn_8_while_identity_4E
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp¢Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_8_while_placeholderVbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_8/while/simple_rnn_cell_26/MatMulMatMulFbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAddBiasAdd?backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul:product:0Mbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1MatMul)backward_simple_rnn_8_while_placeholder_2Nbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_8/while/simple_rnn_cell_26/addAddV2?backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd:output:0Abackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_8/while/simple_rnn_cell_26/TanhTanh6backward_simple_rnn_8/while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_8_while_placeholder_1Obackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_8/while/addAddV2'backward_simple_rnn_8_while_placeholder*backward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_8/while/add_1AddV2Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counter,backward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_8/while/IdentityIdentity%backward_simple_rnn_8/while/add_1:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_8/while/Identity_1IdentityJbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_8/while/Identity_2Identity#backward_simple_rnn_8/while/add:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_8/while/Identity_3IdentityPbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_8/while/Identity_4Identity7backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0!^backward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_8/while/NoOpNoOpF^backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpE^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpG^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0"U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0"Y
&backward_simple_rnn_8_while_identity_1/backward_simple_rnn_8/while/Identity_1:output:0"Y
&backward_simple_rnn_8_while_identity_2/backward_simple_rnn_8/while/Identity_2:output:0"Y
&backward_simple_rnn_8_while_identity_3/backward_simple_rnn_8/while/Identity_3:output:0"Y
&backward_simple_rnn_8_while_identity_4/backward_simple_rnn_8/while/Identity_4:output:0"¢
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resourcePbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resourceQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resourceObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0"
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpEbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpDbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp2
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpFbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 

ì
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4511495

inputs
states_00
matmul_readvariableop_resource:4@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
?
Ê
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510769

inputsC
1simple_rnn_cell_25_matmul_readvariableop_resource:4@@
2simple_rnn_cell_25_biasadd_readvariableop_resource:@E
3simple_rnn_cell_25_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_25/BiasAdd/ReadVariableOp¢(simple_rnn_cell_25/MatMul/ReadVariableOp¢*simple_rnn_cell_25/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿà
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_25/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_25/BiasAddBiasAdd#simple_rnn_cell_25/MatMul:product:01simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_25/MatMul_1MatMulzeros:output:02simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_25/addAddV2#simple_rnn_cell_25/BiasAdd:output:0%simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_25/TanhTanhsimple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_25_matmul_readvariableop_resource2simple_rnn_cell_25_biasadd_readvariableop_resource3simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4510702*
condR
while_cond_4510701*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_25/BiasAdd/ReadVariableOp)^simple_rnn_cell_25/MatMul/ReadVariableOp+^simple_rnn_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_25/BiasAdd/ReadVariableOp)simple_rnn_cell_25/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_25/MatMul/ReadVariableOp(simple_rnn_cell_25/MatMul/ReadVariableOp2X
*simple_rnn_cell_25/MatMul_1/ReadVariableOp*simple_rnn_cell_25/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§Ñ
ï
J__inference_sequential_15_layer_call_and_return_conditional_losses_4509200

inputsi
Wbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource:4@f
Xbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource:@k
Ybidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@j
Xbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource:4@g
Ybidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource:@l
Zbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@:
'dense_15_matmul_readvariableop_resource:	6
(dense_15_biasadd_readvariableop_resource:
identity¢Pbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Obidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp¢Qbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp¢,bidirectional_15/backward_simple_rnn_8/while¢Obidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Nbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp¢Pbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp¢+bidirectional_15/forward_simple_rnn_8/while¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOpa
+bidirectional_15/forward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:
9bidirectional_15/forward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;bidirectional_15/forward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;bidirectional_15/forward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3bidirectional_15/forward_simple_rnn_8/strided_sliceStridedSlice4bidirectional_15/forward_simple_rnn_8/Shape:output:0Bbidirectional_15/forward_simple_rnn_8/strided_slice/stack:output:0Dbidirectional_15/forward_simple_rnn_8/strided_slice/stack_1:output:0Dbidirectional_15/forward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4bidirectional_15/forward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@å
2bidirectional_15/forward_simple_rnn_8/zeros/packedPack<bidirectional_15/forward_simple_rnn_8/strided_slice:output:0=bidirectional_15/forward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:v
1bidirectional_15/forward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Þ
+bidirectional_15/forward_simple_rnn_8/zerosFill;bidirectional_15/forward_simple_rnn_8/zeros/packed:output:0:bidirectional_15/forward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4bidirectional_15/forward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¹
/bidirectional_15/forward_simple_rnn_8/transpose	Transposeinputs=bidirectional_15/forward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
-bidirectional_15/forward_simple_rnn_8/Shape_1Shape3bidirectional_15/forward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:
;bidirectional_15/forward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_15/forward_simple_rnn_8/strided_slice_1StridedSlice6bidirectional_15/forward_simple_rnn_8/Shape_1:output:0Dbidirectional_15/forward_simple_rnn_8/strided_slice_1/stack:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_1:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Abidirectional_15/forward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
3bidirectional_15/forward_simple_rnn_8/TensorArrayV2TensorListReserveJbidirectional_15/forward_simple_rnn_8/TensorArrayV2/element_shape:output:0>bidirectional_15/forward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¬
[bidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ò
Mbidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor3bidirectional_15/forward_simple_rnn_8/transpose:y:0dbidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;bidirectional_15/forward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:§
5bidirectional_15/forward_simple_rnn_8/strided_slice_2StridedSlice3bidirectional_15/forward_simple_rnn_8/transpose:y:0Dbidirectional_15/forward_simple_rnn_8/strided_slice_2/stack:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_1:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskæ
Nbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpWbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
?bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMulMatMul>bidirectional_15/forward_simple_rnn_8/strided_slice_2:output:0Vbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Obidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpXbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¡
@bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAddBiasAddIbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul:product:0Wbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
Pbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpYbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Abidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1MatMul4bidirectional_15/forward_simple_rnn_8/zeros:output:0Xbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
<bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/addAddV2Ibidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd:output:0Kbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¹
=bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/TanhTanh@bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Cbidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Bbidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :·
5bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1TensorListReserveLbidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0Kbidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
*bidirectional_15/forward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 
>bidirectional_15/forward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿz
8bidirectional_15/forward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ë	
+bidirectional_15/forward_simple_rnn_8/whileWhileAbidirectional_15/forward_simple_rnn_8/while/loop_counter:output:0Gbidirectional_15/forward_simple_rnn_8/while/maximum_iterations:output:03bidirectional_15/forward_simple_rnn_8/time:output:0>bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1:handle:04bidirectional_15/forward_simple_rnn_8/zeros:output:0>bidirectional_15/forward_simple_rnn_8/strided_slice_1:output:0]bidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Wbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resourceXbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resourceYbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *D
body<R:
8bidirectional_15_forward_simple_rnn_8_while_body_4509016*D
cond<R:
8bidirectional_15_forward_simple_rnn_8_while_cond_4509015*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations §
Vbidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   È
Hbidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack4bidirectional_15/forward_simple_rnn_8/while:output:3_bidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
;bidirectional_15/forward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
=bidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Å
5bidirectional_15/forward_simple_rnn_8/strided_slice_3StridedSliceQbidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0Dbidirectional_15/forward_simple_rnn_8/strided_slice_3/stack:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_1:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
6bidirectional_15/forward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
1bidirectional_15/forward_simple_rnn_8/transpose_1	TransposeQbidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0?bidirectional_15/forward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
,bidirectional_15/backward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:
:bidirectional_15/backward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<bidirectional_15/backward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<bidirectional_15/backward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4bidirectional_15/backward_simple_rnn_8/strided_sliceStridedSlice5bidirectional_15/backward_simple_rnn_8/Shape:output:0Cbidirectional_15/backward_simple_rnn_8/strided_slice/stack:output:0Ebidirectional_15/backward_simple_rnn_8/strided_slice/stack_1:output:0Ebidirectional_15/backward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5bidirectional_15/backward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@è
3bidirectional_15/backward_simple_rnn_8/zeros/packedPack=bidirectional_15/backward_simple_rnn_8/strided_slice:output:0>bidirectional_15/backward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:w
2bidirectional_15/backward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    á
,bidirectional_15/backward_simple_rnn_8/zerosFill<bidirectional_15/backward_simple_rnn_8/zeros/packed:output:0;bidirectional_15/backward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5bidirectional_15/backward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
0bidirectional_15/backward_simple_rnn_8/transpose	Transposeinputs>bidirectional_15/backward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
.bidirectional_15/backward_simple_rnn_8/Shape_1Shape4bidirectional_15/backward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:
<bidirectional_15/backward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_15/backward_simple_rnn_8/strided_slice_1StridedSlice7bidirectional_15/backward_simple_rnn_8/Shape_1:output:0Ebidirectional_15/backward_simple_rnn_8/strided_slice_1/stack:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_1:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bbidirectional_15/backward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
4bidirectional_15/backward_simple_rnn_8/TensorArrayV2TensorListReserveKbidirectional_15/backward_simple_rnn_8/TensorArrayV2/element_shape:output:0?bidirectional_15/backward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5bidirectional_15/backward_simple_rnn_8/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: é
0bidirectional_15/backward_simple_rnn_8/ReverseV2	ReverseV24bidirectional_15/backward_simple_rnn_8/transpose:y:0>bidirectional_15/backward_simple_rnn_8/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4­
\bidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ú
Nbidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor9bidirectional_15/backward_simple_rnn_8/ReverseV2:output:0ebidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<bidirectional_15/backward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
6bidirectional_15/backward_simple_rnn_8/strided_slice_2StridedSlice4bidirectional_15/backward_simple_rnn_8/transpose:y:0Ebidirectional_15/backward_simple_rnn_8/strided_slice_2/stack:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_1:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskè
Obidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpXbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
@bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMulMatMul?bidirectional_15/backward_simple_rnn_8/strided_slice_2:output:0Wbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Pbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpYbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Abidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAddBiasAddJbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul:product:0Xbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ì
Qbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpZbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Bbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1MatMul5bidirectional_15/backward_simple_rnn_8/zeros:output:0Ybidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
=bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/addAddV2Jbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd:output:0Lbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
>bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/TanhTanhAbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dbidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Cbidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :º
6bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1TensorListReserveMbidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0Lbidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
+bidirectional_15/backward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 
?bidirectional_15/backward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ{
9bidirectional_15/backward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø	
,bidirectional_15/backward_simple_rnn_8/whileWhileBbidirectional_15/backward_simple_rnn_8/while/loop_counter:output:0Hbidirectional_15/backward_simple_rnn_8/while/maximum_iterations:output:04bidirectional_15/backward_simple_rnn_8/time:output:0?bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1:handle:05bidirectional_15/backward_simple_rnn_8/zeros:output:0?bidirectional_15/backward_simple_rnn_8/strided_slice_1:output:0^bidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Xbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resourceYbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resourceZbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *E
body=R;
9bidirectional_15_backward_simple_rnn_8_while_body_4509124*E
cond=R;
9bidirectional_15_backward_simple_rnn_8_while_cond_4509123*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¨
Wbidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
Ibidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack5bidirectional_15/backward_simple_rnn_8/while:output:3`bidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
<bidirectional_15/backward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
>bidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ê
6bidirectional_15/backward_simple_rnn_8/strided_slice_3StridedSliceRbidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0Ebidirectional_15/backward_simple_rnn_8/strided_slice_3/stack:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_1:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
7bidirectional_15/backward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
2bidirectional_15/backward_simple_rnn_8/transpose_1	TransposeRbidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional_15/backward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
bidirectional_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
bidirectional_15/concatConcatV2>bidirectional_15/forward_simple_rnn_8/strided_slice_3:output:0?bidirectional_15/backward_simple_rnn_8/strided_slice_3:output:0%bidirectional_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_15/MatMulMatMul bidirectional_15/concat:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
NoOpNoOpQ^bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpP^bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpR^bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp-^bidirectional_15/backward_simple_rnn_8/whileP^bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpO^bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpQ^bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp,^bidirectional_15/forward_simple_rnn_8/while ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2¤
Pbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpPbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp2¢
Obidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpObidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp2¦
Qbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpQbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp2\
,bidirectional_15/backward_simple_rnn_8/while,bidirectional_15/backward_simple_rnn_8/while2¢
Obidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpObidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp2 
Nbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpNbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp2¤
Pbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpPbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp2Z
+bidirectional_15/forward_simple_rnn_8/while+bidirectional_15/forward_simple_rnn_8/while2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ß
¯
while_cond_4510591
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4510591___redundant_placeholder05
1while_while_cond_4510591___redundant_placeholder15
1while_while_cond_4510591___redundant_placeholder25
1while_while_cond_4510591___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
û§
Ï
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4509715
inputs_0X
Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp¢@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp¢backward_simple_rnn_8/while¢>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp¢?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp¢forward_simple_rnn_8/whileR
forward_simple_rnn_8/ShapeShapeinputs_0*
T0*
_output_shapes
:r
(forward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_8/strided_sliceStridedSlice#forward_simple_rnn_8/Shape:output:01forward_simple_rnn_8/strided_slice/stack:output:03forward_simple_rnn_8/strided_slice/stack_1:output:03forward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_8/zeros/packedPack+forward_simple_rnn_8/strided_slice:output:0,forward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_8/zerosFill*forward_simple_rnn_8/zeros/packed:output:0)forward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
forward_simple_rnn_8/transpose	Transposeinputs_0,forward_simple_rnn_8/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn
forward_simple_rnn_8/Shape_1Shape"forward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_8/strided_slice_1StridedSlice%forward_simple_rnn_8/Shape_1:output:03forward_simple_rnn_8/strided_slice_1/stack:output:05forward_simple_rnn_8/strided_slice_1/stack_1:output:05forward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_8/TensorArrayV2TensorListReserve9forward_simple_rnn_8/TensorArrayV2/element_shape:output:0-forward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
<forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_8/transpose:y:0Sforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
$forward_simple_rnn_8/strided_slice_2StridedSlice"forward_simple_rnn_8/transpose:y:03forward_simple_rnn_8/strided_slice_2/stack:output:05forward_simple_rnn_8/strided_slice_2/stack_1:output:05forward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÄ
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_8/simple_rnn_cell_25/MatMulMatMul-forward_simple_rnn_8/strided_slice_2:output:0Eforward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_8/simple_rnn_cell_25/BiasAddBiasAdd8forward_simple_rnn_8/simple_rnn_cell_25/MatMul:product:0Fforward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1MatMul#forward_simple_rnn_8/zeros:output:0Gforward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_8/simple_rnn_cell_25/addAddV28forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd:output:0:forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_8/simple_rnn_cell_25/TanhTanh/forward_simple_rnn_8/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_8/TensorArrayV2_1TensorListReserve;forward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_8/whileWhile0forward_simple_rnn_8/while/loop_counter:output:06forward_simple_rnn_8/while/maximum_iterations:output:0"forward_simple_rnn_8/time:output:0-forward_simple_rnn_8/TensorArrayV2_1:handle:0#forward_simple_rnn_8/zeros:output:0-forward_simple_rnn_8/strided_slice_1:output:0Lforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resourceGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resourceHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'forward_simple_rnn_8_while_body_4509538*3
cond+R)
'forward_simple_rnn_8_while_cond_4509537*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_8/while:output:3Nforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_8/strided_slice_3StridedSlice@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_8/strided_slice_3/stack:output:05forward_simple_rnn_8/strided_slice_3/stack_1:output:05forward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_8/transpose_1	Transpose@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
backward_simple_rnn_8/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)backward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_8/strided_sliceStridedSlice$backward_simple_rnn_8/Shape:output:02backward_simple_rnn_8/strided_slice/stack:output:04backward_simple_rnn_8/strided_slice/stack_1:output:04backward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_8/zeros/packedPack,backward_simple_rnn_8/strided_slice:output:0-backward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_8/zerosFill+backward_simple_rnn_8/zeros/packed:output:0*backward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
backward_simple_rnn_8/transpose	Transposeinputs_0-backward_simple_rnn_8/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
backward_simple_rnn_8/Shape_1Shape#backward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_8/strided_slice_1StridedSlice&backward_simple_rnn_8/Shape_1:output:04backward_simple_rnn_8/strided_slice_1/stack:output:06backward_simple_rnn_8/strided_slice_1/stack_1:output:06backward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_8/TensorArrayV2TensorListReserve:backward_simple_rnn_8/TensorArrayV2/element_shape:output:0.backward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_8/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: È
backward_simple_rnn_8/ReverseV2	ReverseV2#backward_simple_rnn_8/transpose:y:0-backward_simple_rnn_8/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Kbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ§
=backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_8/ReverseV2:output:0Tbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
%backward_simple_rnn_8/strided_slice_2StridedSlice#backward_simple_rnn_8/transpose:y:04backward_simple_rnn_8/strided_slice_2/stack:output:06backward_simple_rnn_8/strided_slice_2/stack_1:output:06backward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_8/simple_rnn_cell_26/MatMulMatMul.backward_simple_rnn_8/strided_slice_2:output:0Fbackward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_8/simple_rnn_cell_26/BiasAddBiasAdd9backward_simple_rnn_8/simple_rnn_cell_26/MatMul:product:0Gbackward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1MatMul$backward_simple_rnn_8/zeros:output:0Hbackward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_8/simple_rnn_cell_26/addAddV29backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd:output:0;backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_8/simple_rnn_cell_26/TanhTanh0backward_simple_rnn_8/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_8/TensorArrayV2_1TensorListReserve<backward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_8/whileWhile1backward_simple_rnn_8/while/loop_counter:output:07backward_simple_rnn_8/while/maximum_iterations:output:0#backward_simple_rnn_8/time:output:0.backward_simple_rnn_8/TensorArrayV2_1:handle:0$backward_simple_rnn_8/zeros:output:0.backward_simple_rnn_8/strided_slice_1:output:0Mbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resourceHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resourceIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *4
body,R*
(backward_simple_rnn_8_while_body_4509646*4
cond,R*
(backward_simple_rnn_8_while_cond_4509645*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_8/while:output:3Obackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_8/strided_slice_3StridedSliceAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_8/strided_slice_3/stack:output:06backward_simple_rnn_8/strided_slice_3/stack_1:output:06backward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_8/transpose_1	TransposeAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_8/strided_slice_3:output:0.backward_simple_rnn_8/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?^backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpA^backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp^backward_simple_rnn_8/while?^forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>^forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp@^forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp^forward_simple_rnn_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp2
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp2:
backward_simple_rnn_8/whilebackward_simple_rnn_8/while2
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp2
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp28
forward_simple_rnn_8/whileforward_simple_rnn_8/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
²
Ñ
(backward_simple_rnn_8_while_cond_4508388H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2J
Fbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4508388___redundant_placeholder0a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4508388___redundant_placeholder1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4508388___redundant_placeholder2a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4508388___redundant_placeholder3(
$backward_simple_rnn_8_while_identity
º
 backward_simple_rnn_8/while/LessLess'backward_simple_rnn_8_while_placeholderFbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_8/while/IdentityIdentity$backward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ð
Ó
"__inference__wrapped_model_4507072
bidirectional_15_inputw
esequential_15_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource:4@t
fsequential_15_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource:@y
gsequential_15_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@x
fsequential_15_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource:4@u
gsequential_15_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource:@z
hsequential_15_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@H
5sequential_15_dense_15_matmul_readvariableop_resource:	D
6sequential_15_dense_15_biasadd_readvariableop_resource:
identity¢^sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢]sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp¢_sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp¢:sequential_15/bidirectional_15/backward_simple_rnn_8/while¢]sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢\sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp¢^sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp¢9sequential_15/bidirectional_15/forward_simple_rnn_8/while¢-sequential_15/dense_15/BiasAdd/ReadVariableOp¢,sequential_15/dense_15/MatMul/ReadVariableOp
9sequential_15/bidirectional_15/forward_simple_rnn_8/ShapeShapebidirectional_15_input*
T0*
_output_shapes
:
Gsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Isequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Isequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Õ
Asequential_15/bidirectional_15/forward_simple_rnn_8/strided_sliceStridedSliceBsequential_15/bidirectional_15/forward_simple_rnn_8/Shape:output:0Psequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice/stack:output:0Rsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice/stack_1:output:0Rsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bsequential_15/bidirectional_15/forward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
@sequential_15/bidirectional_15/forward_simple_rnn_8/zeros/packedPackJsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice:output:0Ksequential_15/bidirectional_15/forward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
?sequential_15/bidirectional_15/forward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
9sequential_15/bidirectional_15/forward_simple_rnn_8/zerosFillIsequential_15/bidirectional_15/forward_simple_rnn_8/zeros/packed:output:0Hsequential_15/bidirectional_15/forward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Bsequential_15/bidirectional_15/forward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          å
=sequential_15/bidirectional_15/forward_simple_rnn_8/transpose	Transposebidirectional_15_inputKsequential_15/bidirectional_15/forward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4¬
;sequential_15/bidirectional_15/forward_simple_rnn_8/Shape_1ShapeAsequential_15/bidirectional_15/forward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:
Isequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ß
Csequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_1StridedSliceDsequential_15/bidirectional_15/forward_simple_rnn_8/Shape_1:output:0Rsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_1/stack:output:0Tsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_1:output:0Tsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Osequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÐ
Asequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2TensorListReserveXsequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2/element_shape:output:0Lsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒº
isequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ü
[sequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorAsequential_15/bidirectional_15/forward_simple_rnn_8/transpose:y:0rsequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:í
Csequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_2StridedSliceAsequential_15/bidirectional_15/forward_simple_rnn_8/transpose:y:0Rsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_2/stack:output:0Tsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_1:output:0Tsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
\sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpesequential_15_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0½
Msequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMulMatMulLsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_2:output:0dsequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
]sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpfsequential_15_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ë
Nsequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAddBiasAddWsequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul:product:0esequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
^sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpgsequential_15_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0·
Osequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1MatMulBsequential_15/bidirectional_15/forward_simple_rnn_8/zeros:output:0fsequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¹
Jsequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/addAddV2Wsequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd:output:0Ysequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Õ
Ksequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/TanhTanhNsequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
Qsequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Psequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :á
Csequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1TensorListReserveZsequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0Ysequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒz
8sequential_15/bidirectional_15/forward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Lsequential_15/bidirectional_15/forward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Fsequential_15/bidirectional_15/forward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
9sequential_15/bidirectional_15/forward_simple_rnn_8/whileWhileOsequential_15/bidirectional_15/forward_simple_rnn_8/while/loop_counter:output:0Usequential_15/bidirectional_15/forward_simple_rnn_8/while/maximum_iterations:output:0Asequential_15/bidirectional_15/forward_simple_rnn_8/time:output:0Lsequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1:handle:0Bsequential_15/bidirectional_15/forward_simple_rnn_8/zeros:output:0Lsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_1:output:0ksequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0esequential_15_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resourcefsequential_15_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resourcegsequential_15_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *R
bodyJRH
Fsequential_15_bidirectional_15_forward_simple_rnn_8_while_body_4506888*R
condJRH
Fsequential_15_bidirectional_15_forward_simple_rnn_8_while_cond_4506887*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations µ
dsequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ò
Vsequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStackBsequential_15/bidirectional_15/forward_simple_rnn_8/while:output:3msequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
Isequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Ksequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Csequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_3StridedSlice_sequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0Rsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_3/stack:output:0Tsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_1:output:0Tsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
Dsequential_15/bidirectional_15/forward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ²
?sequential_15/bidirectional_15/forward_simple_rnn_8/transpose_1	Transpose_sequential_15/bidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0Msequential_15/bidirectional_15/forward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
:sequential_15/bidirectional_15/backward_simple_rnn_8/ShapeShapebidirectional_15_input*
T0*
_output_shapes
:
Hsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ú
Bsequential_15/bidirectional_15/backward_simple_rnn_8/strided_sliceStridedSliceCsequential_15/bidirectional_15/backward_simple_rnn_8/Shape:output:0Qsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice/stack:output:0Ssequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice/stack_1:output:0Ssequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Csequential_15/bidirectional_15/backward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
Asequential_15/bidirectional_15/backward_simple_rnn_8/zeros/packedPackKsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice:output:0Lsequential_15/bidirectional_15/backward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
@sequential_15/bidirectional_15/backward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
:sequential_15/bidirectional_15/backward_simple_rnn_8/zerosFillJsequential_15/bidirectional_15/backward_simple_rnn_8/zeros/packed:output:0Isequential_15/bidirectional_15/backward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Csequential_15/bidirectional_15/backward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ç
>sequential_15/bidirectional_15/backward_simple_rnn_8/transpose	Transposebidirectional_15_inputLsequential_15/bidirectional_15/backward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4®
<sequential_15/bidirectional_15/backward_simple_rnn_8/Shape_1ShapeBsequential_15/bidirectional_15/backward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:
Jsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ä
Dsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_1StridedSliceEsequential_15/bidirectional_15/backward_simple_rnn_8/Shape_1:output:0Ssequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_1/stack:output:0Usequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_1:output:0Usequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Psequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
Bsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2TensorListReserveYsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2/element_shape:output:0Msequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Csequential_15/bidirectional_15/backward_simple_rnn_8/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
>sequential_15/bidirectional_15/backward_simple_rnn_8/ReverseV2	ReverseV2Bsequential_15/bidirectional_15/backward_simple_rnn_8/transpose:y:0Lsequential_15/bidirectional_15/backward_simple_rnn_8/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4»
jsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
\sequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGsequential_15/bidirectional_15/backward_simple_rnn_8/ReverseV2:output:0ssequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Dsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_2StridedSliceBsequential_15/bidirectional_15/backward_simple_rnn_8/transpose:y:0Ssequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_2/stack:output:0Usequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_1:output:0Usequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
]sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpfsequential_15_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0À
Nsequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMulMatMulMsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_2:output:0esequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
^sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpgsequential_15_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Î
Osequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAddBiasAddXsequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul:product:0fsequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
_sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOphsequential_15_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0º
Psequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1MatMulCsequential_15/bidirectional_15/backward_simple_rnn_8/zeros:output:0gsequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
Ksequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/addAddV2Xsequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd:output:0Zsequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@×
Lsequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/TanhTanhOsequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
Rsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Qsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ä
Dsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1TensorListReserve[sequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0Zsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ{
9sequential_15/bidirectional_15/backward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Msequential_15/bidirectional_15/backward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Gsequential_15/bidirectional_15/backward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
:sequential_15/bidirectional_15/backward_simple_rnn_8/whileWhilePsequential_15/bidirectional_15/backward_simple_rnn_8/while/loop_counter:output:0Vsequential_15/bidirectional_15/backward_simple_rnn_8/while/maximum_iterations:output:0Bsequential_15/bidirectional_15/backward_simple_rnn_8/time:output:0Msequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1:handle:0Csequential_15/bidirectional_15/backward_simple_rnn_8/zeros:output:0Msequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_1:output:0lsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0fsequential_15_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resourcegsequential_15_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resourcehsequential_15_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *S
bodyKRI
Gsequential_15_bidirectional_15_backward_simple_rnn_8_while_body_4506996*S
condKRI
Gsequential_15_bidirectional_15_backward_simple_rnn_8_while_cond_4506995*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¶
esequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   õ
Wsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStackCsequential_15/bidirectional_15/backward_simple_rnn_8/while:output:3nsequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
Jsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Lsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_3StridedSlice`sequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0Ssequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_3/stack:output:0Usequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_1:output:0Usequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
Esequential_15/bidirectional_15/backward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
@sequential_15/bidirectional_15/backward_simple_rnn_8/transpose_1	Transpose`sequential_15/bidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0Nsequential_15/bidirectional_15/backward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
*sequential_15/bidirectional_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¿
%sequential_15/bidirectional_15/concatConcatV2Lsequential_15/bidirectional_15/forward_simple_rnn_8/strided_slice_3:output:0Msequential_15/bidirectional_15/backward_simple_rnn_8/strided_slice_3:output:03sequential_15/bidirectional_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,sequential_15/dense_15/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_15_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¿
sequential_15/dense_15/MatMulMatMul.sequential_15/bidirectional_15/concat:output:04sequential_15/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_15/dense_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_15/dense_15/BiasAddBiasAdd'sequential_15/dense_15/MatMul:product:05sequential_15/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_15/dense_15/SoftmaxSoftmax'sequential_15/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_15/dense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
NoOpNoOp_^sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp^^sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp`^sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp;^sequential_15/bidirectional_15/backward_simple_rnn_8/while^^sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp]^sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp_^sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:^sequential_15/bidirectional_15/forward_simple_rnn_8/while.^sequential_15/dense_15/BiasAdd/ReadVariableOp-^sequential_15/dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2À
^sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp^sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp2¾
]sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp]sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp2Â
_sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp_sequential_15/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp2x
:sequential_15/bidirectional_15/backward_simple_rnn_8/while:sequential_15/bidirectional_15/backward_simple_rnn_8/while2¾
]sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp]sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp2¼
\sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp\sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp2À
^sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp^sequential_15/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp2v
9sequential_15/bidirectional_15/forward_simple_rnn_8/while9sequential_15/bidirectional_15/forward_simple_rnn_8/while2^
-sequential_15/dense_15/BiasAdd/ReadVariableOp-sequential_15/dense_15/BiasAdd/ReadVariableOp2\
,sequential_15/dense_15/MatMul/ReadVariableOp,sequential_15/dense_15/MatMul/ReadVariableOp:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_15_input
ä

J__inference_sequential_15_layer_call_and_return_conditional_losses_4508902
bidirectional_15_input*
bidirectional_15_4508883:4@&
bidirectional_15_4508885:@*
bidirectional_15_4508887:@@*
bidirectional_15_4508889:4@&
bidirectional_15_4508891:@*
bidirectional_15_4508893:@@#
dense_15_4508896:	
dense_15_4508898:
identity¢(bidirectional_15/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall
(bidirectional_15/StatefulPartitionedCallStatefulPartitionedCallbidirectional_15_inputbidirectional_15_4508883bidirectional_15_4508885bidirectional_15_4508887bidirectional_15_4508889bidirectional_15_4508891bidirectional_15_4508893*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508758¡
 dense_15/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_15/StatefulPartitionedCall:output:0dense_15_4508896dense_15_4508898*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_4508483x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_15/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_15/StatefulPartitionedCall(bidirectional_15/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_15_input
ÔB
è
(backward_simple_rnn_8_while_body_4508389H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2G
Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0
backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_8_while_identity*
&backward_simple_rnn_8_while_identity_1*
&backward_simple_rnn_8_while_identity_2*
&backward_simple_rnn_8_while_identity_3*
&backward_simple_rnn_8_while_identity_4E
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp¢Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_8_while_placeholderVbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_8/while/simple_rnn_cell_26/MatMulMatMulFbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAddBiasAdd?backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul:product:0Mbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1MatMul)backward_simple_rnn_8_while_placeholder_2Nbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_8/while/simple_rnn_cell_26/addAddV2?backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd:output:0Abackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_8/while/simple_rnn_cell_26/TanhTanh6backward_simple_rnn_8/while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_8_while_placeholder_1Obackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_8/while/addAddV2'backward_simple_rnn_8_while_placeholder*backward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_8/while/add_1AddV2Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counter,backward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_8/while/IdentityIdentity%backward_simple_rnn_8/while/add_1:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_8/while/Identity_1IdentityJbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_8/while/Identity_2Identity#backward_simple_rnn_8/while/add:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_8/while/Identity_3IdentityPbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_8/while/Identity_4Identity7backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0!^backward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_8/while/NoOpNoOpF^backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpE^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpG^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0"U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0"Y
&backward_simple_rnn_8_while_identity_1/backward_simple_rnn_8/while/Identity_1:output:0"Y
&backward_simple_rnn_8_while_identity_2/backward_simple_rnn_8/while/Identity_2:output:0"Y
&backward_simple_rnn_8_while_identity_3/backward_simple_rnn_8/while/Identity_3:output:0"Y
&backward_simple_rnn_8_while_identity_4/backward_simple_rnn_8/while/Identity_4:output:0"¢
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resourcePbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resourceQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resourceObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0"
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpEbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpDbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp2
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpFbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ß
¯
while_cond_4511191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4511191___redundant_placeholder05
1while_while_cond_4511191___redundant_placeholder15
1while_while_cond_4511191___redundant_placeholder25
1while_while_cond_4511191___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
½"
ß
while_body_4507295
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_25_4507317_0:4@0
"while_simple_rnn_cell_25_4507319_0:@4
"while_simple_rnn_cell_25_4507321_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_25_4507317:4@.
 while_simple_rnn_cell_25_4507319:@2
 while_simple_rnn_cell_25_4507321:@@¢0while/simple_rnn_cell_25/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_25_4507317_0"while_simple_rnn_cell_25_4507319_0"while_simple_rnn_cell_25_4507321_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4507242r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_25/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_25_4507317"while_simple_rnn_cell_25_4507317_0"F
 while_simple_rnn_cell_25_4507319"while_simple_rnn_cell_25_4507319_0"F
 while_simple_rnn_cell_25_4507321"while_simple_rnn_cell_25_4507321_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_25/StatefulPartitionedCall0while/simple_rnn_cell_25/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
¥

÷
E__inference_dense_15_layer_call_and_return_conditional_losses_4508483

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
Í
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4510375

inputsX
Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp¢@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp¢backward_simple_rnn_8/while¢>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp¢?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp¢forward_simple_rnn_8/whileP
forward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_8/strided_sliceStridedSlice#forward_simple_rnn_8/Shape:output:01forward_simple_rnn_8/strided_slice/stack:output:03forward_simple_rnn_8/strided_slice/stack_1:output:03forward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_8/zeros/packedPack+forward_simple_rnn_8/strided_slice:output:0,forward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_8/zerosFill*forward_simple_rnn_8/zeros/packed:output:0)forward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_8/transpose	Transposeinputs,forward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4n
forward_simple_rnn_8/Shape_1Shape"forward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_8/strided_slice_1StridedSlice%forward_simple_rnn_8/Shape_1:output:03forward_simple_rnn_8/strided_slice_1/stack:output:05forward_simple_rnn_8/strided_slice_1/stack_1:output:05forward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_8/TensorArrayV2TensorListReserve9forward_simple_rnn_8/TensorArrayV2/element_shape:output:0-forward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
<forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_8/transpose:y:0Sforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$forward_simple_rnn_8/strided_slice_2StridedSlice"forward_simple_rnn_8/transpose:y:03forward_simple_rnn_8/strided_slice_2/stack:output:05forward_simple_rnn_8/strided_slice_2/stack_1:output:05forward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÄ
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_8/simple_rnn_cell_25/MatMulMatMul-forward_simple_rnn_8/strided_slice_2:output:0Eforward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_8/simple_rnn_cell_25/BiasAddBiasAdd8forward_simple_rnn_8/simple_rnn_cell_25/MatMul:product:0Fforward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1MatMul#forward_simple_rnn_8/zeros:output:0Gforward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_8/simple_rnn_cell_25/addAddV28forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd:output:0:forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_8/simple_rnn_cell_25/TanhTanh/forward_simple_rnn_8/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_8/TensorArrayV2_1TensorListReserve;forward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_8/whileWhile0forward_simple_rnn_8/while/loop_counter:output:06forward_simple_rnn_8/while/maximum_iterations:output:0"forward_simple_rnn_8/time:output:0-forward_simple_rnn_8/TensorArrayV2_1:handle:0#forward_simple_rnn_8/zeros:output:0-forward_simple_rnn_8/strided_slice_1:output:0Lforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resourceGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resourceHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'forward_simple_rnn_8_while_body_4510198*3
cond+R)
'forward_simple_rnn_8_while_cond_4510197*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_8/while:output:3Nforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_8/strided_slice_3StridedSlice@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_8/strided_slice_3/stack:output:05forward_simple_rnn_8/strided_slice_3/stack_1:output:05forward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_8/transpose_1	Transpose@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
backward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_8/strided_sliceStridedSlice$backward_simple_rnn_8/Shape:output:02backward_simple_rnn_8/strided_slice/stack:output:04backward_simple_rnn_8/strided_slice/stack_1:output:04backward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_8/zeros/packedPack,backward_simple_rnn_8/strided_slice:output:0-backward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_8/zerosFill+backward_simple_rnn_8/zeros/packed:output:0*backward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_simple_rnn_8/transpose	Transposeinputs-backward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
backward_simple_rnn_8/Shape_1Shape#backward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_8/strided_slice_1StridedSlice&backward_simple_rnn_8/Shape_1:output:04backward_simple_rnn_8/strided_slice_1/stack:output:06backward_simple_rnn_8/strided_slice_1/stack_1:output:06backward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_8/TensorArrayV2TensorListReserve:backward_simple_rnn_8/TensorArrayV2/element_shape:output:0.backward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_8/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¶
backward_simple_rnn_8/ReverseV2	ReverseV2#backward_simple_rnn_8/transpose:y:0-backward_simple_rnn_8/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Kbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   §
=backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_8/ReverseV2:output:0Tbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%backward_simple_rnn_8/strided_slice_2StridedSlice#backward_simple_rnn_8/transpose:y:04backward_simple_rnn_8/strided_slice_2/stack:output:06backward_simple_rnn_8/strided_slice_2/stack_1:output:06backward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_8/simple_rnn_cell_26/MatMulMatMul.backward_simple_rnn_8/strided_slice_2:output:0Fbackward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_8/simple_rnn_cell_26/BiasAddBiasAdd9backward_simple_rnn_8/simple_rnn_cell_26/MatMul:product:0Gbackward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1MatMul$backward_simple_rnn_8/zeros:output:0Hbackward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_8/simple_rnn_cell_26/addAddV29backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd:output:0;backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_8/simple_rnn_cell_26/TanhTanh0backward_simple_rnn_8/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_8/TensorArrayV2_1TensorListReserve<backward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_8/whileWhile1backward_simple_rnn_8/while/loop_counter:output:07backward_simple_rnn_8/while/maximum_iterations:output:0#backward_simple_rnn_8/time:output:0.backward_simple_rnn_8/TensorArrayV2_1:handle:0$backward_simple_rnn_8/zeros:output:0.backward_simple_rnn_8/strided_slice_1:output:0Mbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resourceHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resourceIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *4
body,R*
(backward_simple_rnn_8_while_body_4510306*4
cond,R*
(backward_simple_rnn_8_while_cond_4510305*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_8/while:output:3Obackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_8/strided_slice_3StridedSliceAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_8/strided_slice_3/stack:output:06backward_simple_rnn_8/strided_slice_3/stack_1:output:06backward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_8/transpose_1	TransposeAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_8/strided_slice_3:output:0.backward_simple_rnn_8/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?^backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpA^backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp^backward_simple_rnn_8/while?^forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>^forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp@^forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp^forward_simple_rnn_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp2
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp2:
backward_simple_rnn_8/whilebackward_simple_rnn_8/while2
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp2
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp28
forward_simple_rnn_8/whileforward_simple_rnn_8/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ü-
Ò
while_body_4510702
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_25_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_25/MatMul/ReadVariableOp¢0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_25/BiasAddBiasAdd)while/simple_rnn_cell_25/MatMul:product:07while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_25/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_25/addAddV2)while/simple_rnn_cell_25/BiasAdd:output:0+while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_25/TanhTanh while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_25/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_25/MatMul/ReadVariableOp1^while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_25_matmul_readvariableop_resource9while_simple_rnn_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_25/MatMul/ReadVariableOp.while/simple_rnn_cell_25/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
²
Ñ
(backward_simple_rnn_8_while_cond_4508688H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2J
Fbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4508688___redundant_placeholder0a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4508688___redundant_placeholder1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4508688___redundant_placeholder2a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4508688___redundant_placeholder3(
$backward_simple_rnn_8_while_identity
º
 backward_simple_rnn_8/while/LessLess'backward_simple_rnn_8_while_placeholderFbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_8/while/IdentityIdentity$backward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Ù@
Í
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511035
inputs_0C
1simple_rnn_cell_26_matmul_readvariableop_resource:4@@
2simple_rnn_cell_26_biasadd_readvariableop_resource:@E
3simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_26/BiasAdd/ReadVariableOp¢(simple_rnn_cell_26/MatMul/ReadVariableOp¢*simple_rnn_cell_26/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
(simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_26/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_26/BiasAddBiasAdd#simple_rnn_cell_26/MatMul:product:01simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_26/MatMul_1MatMulzeros:output:02simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_26/addAddV2#simple_rnn_cell_26/BiasAdd:output:0%simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_26/TanhTanhsimple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_26_matmul_readvariableop_resource2simple_rnn_cell_26_biasadd_readvariableop_resource3simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4510968*
condR
while_cond_4510967*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_26/BiasAdd/ReadVariableOp)^simple_rnn_cell_26/MatMul/ReadVariableOp+^simple_rnn_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_26/BiasAdd/ReadVariableOp)simple_rnn_cell_26/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_26/MatMul/ReadVariableOp(simple_rnn_cell_26/MatMul/ReadVariableOp2X
*simple_rnn_cell_26/MatMul_1/ReadVariableOp*simple_rnn_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
¹
Á
7__inference_backward_simple_rnn_8_layer_call_fn_4510923

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4508054o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

Ü
4__inference_simple_rnn_cell_26_layer_call_fn_4511447

inputs
states_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4507418o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
ß
¯
while_cond_4511303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4511303___redundant_placeholder05
1while_while_cond_4511303___redundant_placeholder15
1while_while_cond_4511303___redundant_placeholder25
1while_while_cond_4511303___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ß
¯
while_cond_4507133
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4507133___redundant_placeholder05
1while_while_cond_4507133___redundant_placeholder15
1while_while_cond_4507133___redundant_placeholder25
1while_while_cond_4507133___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
²
Ñ
(backward_simple_rnn_8_while_cond_4510305H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2J
Fbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4510305___redundant_placeholder0a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4510305___redundant_placeholder1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4510305___redundant_placeholder2a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4510305___redundant_placeholder3(
$backward_simple_rnn_8_while_identity
º
 backward_simple_rnn_8/while/LessLess'backward_simple_rnn_8_while_placeholderFbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_8/while/IdentityIdentity$backward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ß
¯
while_cond_4511079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4511079___redundant_placeholder05
1while_while_cond_4511079___redundant_placeholder15
1while_while_cond_4511079___redundant_placeholder25
1while_while_cond_4511079___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
éR
ì
 __inference__traced_save_4511617
file_prefix.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop^
Zsavev2_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_read_readvariableoph
dsavev2_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_read_readvariableop\
Xsavev2_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_read_readvariableop_
[savev2_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_read_readvariableopi
esavev2_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_read_readvariableop]
Ysavev2_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableope
asavev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_m_read_readvariableopo
ksavev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_m_read_readvariableopc
_savev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_m_read_readvariableopf
bsavev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_m_read_readvariableopp
lsavev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_m_read_readvariableopd
`savev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableope
asavev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_v_read_readvariableopo
ksavev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_v_read_readvariableopc
_savev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_v_read_readvariableopf
bsavev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_v_read_readvariableopp
lsavev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_v_read_readvariableopd
`savev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
: ¡
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ê
valueÀB½"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ì
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableopZsavev2_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_read_readvariableopdsavev2_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_read_readvariableopXsavev2_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_read_readvariableop[savev2_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_read_readvariableopesavev2_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_read_readvariableopYsavev2_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableopasavev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_m_read_readvariableopksavev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_m_read_readvariableop_savev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_m_read_readvariableopbsavev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_m_read_readvariableoplsavev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_m_read_readvariableop`savev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableopasavev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_v_read_readvariableopksavev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_v_read_readvariableop_savev2_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_v_read_readvariableopbsavev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_v_read_readvariableoplsavev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_v_read_readvariableop`savev2_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
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

identity_1Identity_1:output:0*ú
_input_shapesè
å: :	::4@:@@:@:4@:@@:@: : : : : : : : : :	::4@:@@:@:4@:@@:@:	::4@:@@:@:4@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:4@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:4@:$ 

_output_shapes

:@@: 

_output_shapes
:@:	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:4@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:4@:$ 

_output_shapes

:@@: 

_output_shapes
:@:%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:4@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:4@:$  

_output_shapes

:@@: !

_output_shapes
:@:"

_output_shapes
: 
ÙQ
Ê
8bidirectional_15_forward_simple_rnn_8_while_body_4509243h
dbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_loop_countern
jbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_maximum_iterations;
7bidirectional_15_forward_simple_rnn_8_while_placeholder=
9bidirectional_15_forward_simple_rnn_8_while_placeholder_1=
9bidirectional_15_forward_simple_rnn_8_while_placeholder_2g
cbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_strided_slice_1_0¤
bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0q
_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@n
`bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@s
abidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@8
4bidirectional_15_forward_simple_rnn_8_while_identity:
6bidirectional_15_forward_simple_rnn_8_while_identity_1:
6bidirectional_15_forward_simple_rnn_8_while_identity_2:
6bidirectional_15_forward_simple_rnn_8_while_identity_3:
6bidirectional_15_forward_simple_rnn_8_while_identity_4e
abidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_strided_slice_1¢
bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensoro
]bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource:4@l
^bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource:@q
_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢Ubidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Tbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp¢Vbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp®
]bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   å
Obidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_07bidirectional_15_forward_simple_rnn_8_while_placeholderfbidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ô
Tbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0·
Ebidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMulMatMulVbidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0\bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ò
Ubidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp`bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0³
Fbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAddBiasAddObidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul:product:0]bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ø
Vbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpabidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0
Gbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1MatMul9bidirectional_15_forward_simple_rnn_8_while_placeholder_2^bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
Bbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/addAddV2Obidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd:output:0Qbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
Cbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/TanhTanhFbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Vbidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Pbidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem9bidirectional_15_forward_simple_rnn_8_while_placeholder_1_bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:0Gbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒs
1bidirectional_15/forward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Î
/bidirectional_15/forward_simple_rnn_8/while/addAddV27bidirectional_15_forward_simple_rnn_8_while_placeholder:bidirectional_15/forward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: u
3bidirectional_15/forward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ÿ
1bidirectional_15/forward_simple_rnn_8/while/add_1AddV2dbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_loop_counter<bidirectional_15/forward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: Ë
4bidirectional_15/forward_simple_rnn_8/while/IdentityIdentity5bidirectional_15/forward_simple_rnn_8/while/add_1:z:01^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
6bidirectional_15/forward_simple_rnn_8/while/Identity_1Identityjbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_maximum_iterations1^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Ë
6bidirectional_15/forward_simple_rnn_8/while/Identity_2Identity3bidirectional_15/forward_simple_rnn_8/while/add:z:01^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ø
6bidirectional_15/forward_simple_rnn_8/while/Identity_3Identity`bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ð
6bidirectional_15/forward_simple_rnn_8/while/Identity_4IdentityGbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:01^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
0bidirectional_15/forward_simple_rnn_8/while/NoOpNoOpV^bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpU^bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpW^bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "È
abidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_strided_slice_1cbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_strided_slice_1_0"u
4bidirectional_15_forward_simple_rnn_8_while_identity=bidirectional_15/forward_simple_rnn_8/while/Identity:output:0"y
6bidirectional_15_forward_simple_rnn_8_while_identity_1?bidirectional_15/forward_simple_rnn_8/while/Identity_1:output:0"y
6bidirectional_15_forward_simple_rnn_8_while_identity_2?bidirectional_15/forward_simple_rnn_8/while/Identity_2:output:0"y
6bidirectional_15_forward_simple_rnn_8_while_identity_3?bidirectional_15/forward_simple_rnn_8/while/Identity_3:output:0"y
6bidirectional_15_forward_simple_rnn_8_while_identity_4?bidirectional_15/forward_simple_rnn_8/while/Identity_4:output:0"Â
^bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource`bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"Ä
_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resourceabidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"À
]bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0"Â
bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorbidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2®
Ubidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpUbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2¬
Tbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpTbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp2°
Vbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpVbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
õ_

Gsequential_15_bidirectional_15_backward_simple_rnn_8_while_body_4506996
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_loop_counter
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_maximum_iterationsJ
Fsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholderL
Hsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholder_1L
Hsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholder_2
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_strided_slice_1_0Â
½sequential_15_bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0
nsequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@}
osequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@
psequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@G
Csequential_15_bidirectional_15_backward_simple_rnn_8_while_identityI
Esequential_15_bidirectional_15_backward_simple_rnn_8_while_identity_1I
Esequential_15_bidirectional_15_backward_simple_rnn_8_while_identity_2I
Esequential_15_bidirectional_15_backward_simple_rnn_8_while_identity_3I
Esequential_15_bidirectional_15_backward_simple_rnn_8_while_identity_4
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_strided_slice_1À
»sequential_15_bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor~
lsequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource:4@{
msequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource:@
nsequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢dsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢csequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp¢esequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp½
lsequential_15/bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   °
^sequential_15/bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem½sequential_15_bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0Fsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholderusequential_15/bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0
csequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpnsequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0ä
Tsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMulMatMulesequential_15/bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0ksequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOposequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0à
Usequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAddBiasAdd^sequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul:product:0lsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
esequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOppsequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Ë
Vsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1MatMulHsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholder_2msequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
Qsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/addAddV2^sequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd:output:0`sequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ã
Rsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/TanhTanhUsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
esequential_15/bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
_sequential_15/bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemHsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholder_1nsequential_15/bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:0Vsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ
@sequential_15/bidirectional_15/backward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :û
>sequential_15/bidirectional_15/backward_simple_rnn_8/while/addAddV2Fsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholderIsequential_15/bidirectional_15/backward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: 
Bsequential_15/bidirectional_15/backward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¼
@sequential_15/bidirectional_15/backward_simple_rnn_8/while/add_1AddV2sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_loop_counterKsequential_15/bidirectional_15/backward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: ø
Csequential_15/bidirectional_15/backward_simple_rnn_8/while/IdentityIdentityDsequential_15/bidirectional_15/backward_simple_rnn_8/while/add_1:z:0@^sequential_15/bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ¿
Esequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity_1Identitysequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_maximum_iterations@^sequential_15/bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ø
Esequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity_2IdentityBsequential_15/bidirectional_15/backward_simple_rnn_8/while/add:z:0@^sequential_15/bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ¥
Esequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity_3Identityosequential_15/bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0@^sequential_15/bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
Esequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity_4IdentityVsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0@^sequential_15/bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
?sequential_15/bidirectional_15/backward_simple_rnn_8/while/NoOpNoOpe^sequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpd^sequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpf^sequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Csequential_15_bidirectional_15_backward_simple_rnn_8_while_identityLsequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity:output:0"
Esequential_15_bidirectional_15_backward_simple_rnn_8_while_identity_1Nsequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity_1:output:0"
Esequential_15_bidirectional_15_backward_simple_rnn_8_while_identity_2Nsequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity_2:output:0"
Esequential_15_bidirectional_15_backward_simple_rnn_8_while_identity_3Nsequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity_3:output:0"
Esequential_15_bidirectional_15_backward_simple_rnn_8_while_identity_4Nsequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity_4:output:0"
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_strided_slice_1sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_strided_slice_1_0"à
msequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resourceosequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"â
nsequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resourcepsequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0"Þ
lsequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resourcensequential_15_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0"þ
»sequential_15_bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor½sequential_15_bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2Ì
dsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpdsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2Ê
csequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpcsequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp2Î
esequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpesequential_15/bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
·
À
6__inference_forward_simple_rnn_8_layer_call_fn_4510439

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4508186o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
'forward_simple_rnn_8_while_cond_4510197F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2H
Dforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4510197___redundant_placeholder0_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4510197___redundant_placeholder1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4510197___redundant_placeholder2_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4510197___redundant_placeholder3'
#forward_simple_rnn_8_while_identity
¶
forward_simple_rnn_8/while/LessLess&forward_simple_rnn_8_while_placeholderDforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_8/while/IdentityIdentity#forward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
­
Ã
7__inference_backward_simple_rnn_8_layer_call_fn_4510901
inputs_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4507659o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
5
«
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4507359

inputs,
simple_rnn_cell_25_4507282:4@(
simple_rnn_cell_25_4507284:@,
simple_rnn_cell_25_4507286:@@
identity¢*simple_rnn_cell_25/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskó
*simple_rnn_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_25_4507282simple_rnn_cell_25_4507284simple_rnn_cell_25_4507286*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4507242n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_25_4507282simple_rnn_cell_25_4507284simple_rnn_cell_25_4507286*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4507295*
condR
while_cond_4507294*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
NoOpNoOp+^simple_rnn_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_25/StatefulPartitionedCall*simple_rnn_cell_25/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
²
Ñ
(backward_simple_rnn_8_while_cond_4509645H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2J
Fbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4509645___redundant_placeholder0a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4509645___redundant_placeholder1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4509645___redundant_placeholder2a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4509645___redundant_placeholder3(
$backward_simple_rnn_8_while_identity
º
 backward_simple_rnn_8/while/LessLess'backward_simple_rnn_8_while_placeholderFbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_8/while/IdentityIdentity$backward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
5
«
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4507198

inputs,
simple_rnn_cell_25_4507121:4@(
simple_rnn_cell_25_4507123:@,
simple_rnn_cell_25_4507125:@@
identity¢*simple_rnn_cell_25/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskó
*simple_rnn_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_25_4507121simple_rnn_cell_25_4507123simple_rnn_cell_25_4507125*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4507120n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_25_4507121simple_rnn_cell_25_4507123simple_rnn_cell_25_4507125*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4507134*
condR
while_cond_4507133*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
NoOpNoOp+^simple_rnn_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_25/StatefulPartitionedCall*simple_rnn_cell_25/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
þ6
¬
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4507659

inputs,
simple_rnn_cell_26_4507582:4@(
simple_rnn_cell_26_4507584:@,
simple_rnn_cell_26_4507586:@@
identity¢*simple_rnn_cell_26/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskó
*simple_rnn_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_26_4507582simple_rnn_cell_26_4507584simple_rnn_cell_26_4507586*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4507540n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_26_4507582simple_rnn_cell_26_4507584simple_rnn_cell_26_4507586*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4507595*
condR
while_cond_4507594*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
NoOpNoOp+^simple_rnn_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_26/StatefulPartitionedCall*simple_rnn_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
æ
æ
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4507914

inputs.
forward_simple_rnn_8_4507785:4@*
forward_simple_rnn_8_4507787:@.
forward_simple_rnn_8_4507789:@@/
backward_simple_rnn_8_4507904:4@+
backward_simple_rnn_8_4507906:@/
backward_simple_rnn_8_4507908:@@
identity¢-backward_simple_rnn_8/StatefulPartitionedCall¢,forward_simple_rnn_8/StatefulPartitionedCallÆ
,forward_simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_8_4507785forward_simple_rnn_8_4507787forward_simple_rnn_8_4507789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4507784Ë
-backward_simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_8_4507904backward_simple_rnn_8_4507906backward_simple_rnn_8_4507908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4507903M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ó
concatConcatV25forward_simple_rnn_8/StatefulPartitionedCall:output:06backward_simple_rnn_8/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp.^backward_simple_rnn_8/StatefulPartitionedCall-^forward_simple_rnn_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_simple_rnn_8/StatefulPartitionedCall-backward_simple_rnn_8/StatefulPartitionedCall2\
,forward_simple_rnn_8/StatefulPartitionedCall,forward_simple_rnn_8/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Ó
/__inference_sequential_15_layer_call_fn_4508509
bidirectional_15_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508490o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_15_input
ü-
Ò
while_body_4507836
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_26_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_26/MatMul/ReadVariableOp¢0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_26/BiasAddBiasAdd)while/simple_rnn_cell_26/MatMul:product:07while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_26/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_26/addAddV2)while/simple_rnn_cell_26/BiasAdd:output:0+while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_26/TanhTanh while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_26/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_26/MatMul/ReadVariableOp1^while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_26_matmul_readvariableop_resource9while_simple_rnn_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_26/MatMul/ReadVariableOp.while/simple_rnn_cell_26/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
§
Í
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4510155

inputsX
Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp¢@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp¢backward_simple_rnn_8/while¢>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp¢?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp¢forward_simple_rnn_8/whileP
forward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_8/strided_sliceStridedSlice#forward_simple_rnn_8/Shape:output:01forward_simple_rnn_8/strided_slice/stack:output:03forward_simple_rnn_8/strided_slice/stack_1:output:03forward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_8/zeros/packedPack+forward_simple_rnn_8/strided_slice:output:0,forward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_8/zerosFill*forward_simple_rnn_8/zeros/packed:output:0)forward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_8/transpose	Transposeinputs,forward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4n
forward_simple_rnn_8/Shape_1Shape"forward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_8/strided_slice_1StridedSlice%forward_simple_rnn_8/Shape_1:output:03forward_simple_rnn_8/strided_slice_1/stack:output:05forward_simple_rnn_8/strided_slice_1/stack_1:output:05forward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_8/TensorArrayV2TensorListReserve9forward_simple_rnn_8/TensorArrayV2/element_shape:output:0-forward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
<forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_8/transpose:y:0Sforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$forward_simple_rnn_8/strided_slice_2StridedSlice"forward_simple_rnn_8/transpose:y:03forward_simple_rnn_8/strided_slice_2/stack:output:05forward_simple_rnn_8/strided_slice_2/stack_1:output:05forward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÄ
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_8/simple_rnn_cell_25/MatMulMatMul-forward_simple_rnn_8/strided_slice_2:output:0Eforward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_8/simple_rnn_cell_25/BiasAddBiasAdd8forward_simple_rnn_8/simple_rnn_cell_25/MatMul:product:0Fforward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1MatMul#forward_simple_rnn_8/zeros:output:0Gforward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_8/simple_rnn_cell_25/addAddV28forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd:output:0:forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_8/simple_rnn_cell_25/TanhTanh/forward_simple_rnn_8/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_8/TensorArrayV2_1TensorListReserve;forward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_8/whileWhile0forward_simple_rnn_8/while/loop_counter:output:06forward_simple_rnn_8/while/maximum_iterations:output:0"forward_simple_rnn_8/time:output:0-forward_simple_rnn_8/TensorArrayV2_1:handle:0#forward_simple_rnn_8/zeros:output:0-forward_simple_rnn_8/strided_slice_1:output:0Lforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resourceGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resourceHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'forward_simple_rnn_8_while_body_4509978*3
cond+R)
'forward_simple_rnn_8_while_cond_4509977*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_8/while:output:3Nforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_8/strided_slice_3StridedSlice@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_8/strided_slice_3/stack:output:05forward_simple_rnn_8/strided_slice_3/stack_1:output:05forward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_8/transpose_1	Transpose@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
backward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_8/strided_sliceStridedSlice$backward_simple_rnn_8/Shape:output:02backward_simple_rnn_8/strided_slice/stack:output:04backward_simple_rnn_8/strided_slice/stack_1:output:04backward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_8/zeros/packedPack,backward_simple_rnn_8/strided_slice:output:0-backward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_8/zerosFill+backward_simple_rnn_8/zeros/packed:output:0*backward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_simple_rnn_8/transpose	Transposeinputs-backward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
backward_simple_rnn_8/Shape_1Shape#backward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_8/strided_slice_1StridedSlice&backward_simple_rnn_8/Shape_1:output:04backward_simple_rnn_8/strided_slice_1/stack:output:06backward_simple_rnn_8/strided_slice_1/stack_1:output:06backward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_8/TensorArrayV2TensorListReserve:backward_simple_rnn_8/TensorArrayV2/element_shape:output:0.backward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_8/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¶
backward_simple_rnn_8/ReverseV2	ReverseV2#backward_simple_rnn_8/transpose:y:0-backward_simple_rnn_8/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Kbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   §
=backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_8/ReverseV2:output:0Tbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%backward_simple_rnn_8/strided_slice_2StridedSlice#backward_simple_rnn_8/transpose:y:04backward_simple_rnn_8/strided_slice_2/stack:output:06backward_simple_rnn_8/strided_slice_2/stack_1:output:06backward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_8/simple_rnn_cell_26/MatMulMatMul.backward_simple_rnn_8/strided_slice_2:output:0Fbackward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_8/simple_rnn_cell_26/BiasAddBiasAdd9backward_simple_rnn_8/simple_rnn_cell_26/MatMul:product:0Gbackward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1MatMul$backward_simple_rnn_8/zeros:output:0Hbackward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_8/simple_rnn_cell_26/addAddV29backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd:output:0;backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_8/simple_rnn_cell_26/TanhTanh0backward_simple_rnn_8/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_8/TensorArrayV2_1TensorListReserve<backward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_8/whileWhile1backward_simple_rnn_8/while/loop_counter:output:07backward_simple_rnn_8/while/maximum_iterations:output:0#backward_simple_rnn_8/time:output:0.backward_simple_rnn_8/TensorArrayV2_1:handle:0$backward_simple_rnn_8/zeros:output:0.backward_simple_rnn_8/strided_slice_1:output:0Mbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resourceHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resourceIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *4
body,R*
(backward_simple_rnn_8_while_body_4510086*4
cond,R*
(backward_simple_rnn_8_while_cond_4510085*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_8/while:output:3Obackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_8/strided_slice_3StridedSliceAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_8/strided_slice_3/stack:output:06backward_simple_rnn_8/strided_slice_3/stack_1:output:06backward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_8/transpose_1	TransposeAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_8/strided_slice_3:output:0.backward_simple_rnn_8/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?^backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpA^backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp^backward_simple_rnn_8/while?^forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>^forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp@^forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp^forward_simple_rnn_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp2
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp2:
backward_simple_rnn_8/whilebackward_simple_rnn_8/while2
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp2
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp28
forward_simple_rnn_8/whileforward_simple_rnn_8/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ü-
Ò
while_body_4508119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_25_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_25/MatMul/ReadVariableOp¢0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_25/BiasAddBiasAdd)while/simple_rnn_cell_25/MatMul:product:07while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_25/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_25/addAddV2)while/simple_rnn_cell_25/BiasAdd:output:0+while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_25/TanhTanh while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_25/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_25/MatMul/ReadVariableOp1^while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_25_matmul_readvariableop_resource9while_simple_rnn_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_25/MatMul/ReadVariableOp.while/simple_rnn_cell_25/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ÛA
É
'forward_simple_rnn_8_while_body_4509978F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2E
Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0
}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_8_while_identity)
%forward_simple_rnn_8_while_identity_1)
%forward_simple_rnn_8_while_identity_2)
%forward_simple_rnn_8_while_identity_3)
%forward_simple_rnn_8_while_identity_4C
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp¢Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
Lforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
>forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_8_while_placeholderUforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ò
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_8/while/simple_rnn_cell_25/MatMulMatMulEforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAddBiasAdd>forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul:product:0Lforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1MatMul(forward_simple_rnn_8_while_placeholder_2Mforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_8/while/simple_rnn_cell_25/addAddV2>forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd:output:0@forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_8/while/simple_rnn_cell_25/TanhTanh5forward_simple_rnn_8/while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_8_while_placeholder_1Nforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_8/while/addAddV2&forward_simple_rnn_8_while_placeholder)forward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_8/while/add_1AddV2Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counter+forward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_8/while/IdentityIdentity$forward_simple_rnn_8/while/add_1:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_8/while/Identity_1IdentityHforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_8/while/Identity_2Identity"forward_simple_rnn_8/while/add:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_8/while/Identity_3IdentityOforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_8/while/Identity_4Identity6forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0 ^forward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_8/while/NoOpNoOpE^forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpD^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpF^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0"S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0"W
%forward_simple_rnn_8_while_identity_1.forward_simple_rnn_8/while/Identity_1:output:0"W
%forward_simple_rnn_8_while_identity_2.forward_simple_rnn_8/while/Identity_2:output:0"W
%forward_simple_rnn_8_while_identity_3.forward_simple_rnn_8/while/Identity_3:output:0"W
%forward_simple_rnn_8_while_identity_4.forward_simple_rnn_8/while/Identity_4:output:0" 
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resourceOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resourcePforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resourceNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpDforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpCforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp2
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpEforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ÿ@
Ë
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511371

inputsC
1simple_rnn_cell_26_matmul_readvariableop_resource:4@@
2simple_rnn_cell_26_biasadd_readvariableop_resource:@E
3simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_26/BiasAdd/ReadVariableOp¢(simple_rnn_cell_26/MatMul/ReadVariableOp¢*simple_rnn_cell_26/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿå
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_26/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_26/BiasAddBiasAdd#simple_rnn_cell_26/MatMul:product:01simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_26/MatMul_1MatMulzeros:output:02simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_26/addAddV2#simple_rnn_cell_26/BiasAdd:output:0%simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_26/TanhTanhsimple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_26_matmul_readvariableop_resource2simple_rnn_cell_26_biasadd_readvariableop_resource3simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4511304*
condR
while_cond_4511303*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_26/BiasAdd/ReadVariableOp)^simple_rnn_cell_26/MatMul/ReadVariableOp+^simple_rnn_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_26/BiasAdd/ReadVariableOp)simple_rnn_cell_26/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_26/MatMul/ReadVariableOp(simple_rnn_cell_26/MatMul/ReadVariableOp2X
*simple_rnn_cell_26/MatMul_1/ReadVariableOp*simple_rnn_cell_26/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü-
Ò
while_body_4511192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_26_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_26/MatMul/ReadVariableOp¢0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_26/BiasAddBiasAdd)while/simple_rnn_cell_26/MatMul:product:07while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_26/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_26/addAddV2)while/simple_rnn_cell_26/BiasAdd:output:0+while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_26/TanhTanh while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_26/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_26/MatMul/ReadVariableOp1^while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_26_matmul_readvariableop_resource9while_simple_rnn_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_26/MatMul/ReadVariableOp.while/simple_rnn_cell_26/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ç
	
8bidirectional_15_forward_simple_rnn_8_while_cond_4509242h
dbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_loop_countern
jbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_maximum_iterations;
7bidirectional_15_forward_simple_rnn_8_while_placeholder=
9bidirectional_15_forward_simple_rnn_8_while_placeholder_1=
9bidirectional_15_forward_simple_rnn_8_while_placeholder_2j
fbidirectional_15_forward_simple_rnn_8_while_less_bidirectional_15_forward_simple_rnn_8_strided_slice_1
}bidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_cond_4509242___redundant_placeholder0
}bidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_cond_4509242___redundant_placeholder1
}bidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_cond_4509242___redundant_placeholder2
}bidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_cond_4509242___redundant_placeholder38
4bidirectional_15_forward_simple_rnn_8_while_identity
ú
0bidirectional_15/forward_simple_rnn_8/while/LessLess7bidirectional_15_forward_simple_rnn_8_while_placeholderfbidirectional_15_forward_simple_rnn_8_while_less_bidirectional_15_forward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: 
4bidirectional_15/forward_simple_rnn_8/while/IdentityIdentity4bidirectional_15/forward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "u
4bidirectional_15_forward_simple_rnn_8_while_identity=bidirectional_15/forward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ú>
Ì
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510659
inputs_0C
1simple_rnn_cell_25_matmul_readvariableop_resource:4@@
2simple_rnn_cell_25_biasadd_readvariableop_resource:@E
3simple_rnn_cell_25_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_25/BiasAdd/ReadVariableOp¢(simple_rnn_cell_25/MatMul/ReadVariableOp¢*simple_rnn_cell_25/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
(simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_25/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_25/BiasAddBiasAdd#simple_rnn_cell_25/MatMul:product:01simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_25/MatMul_1MatMulzeros:output:02simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_25/addAddV2#simple_rnn_cell_25/BiasAdd:output:0%simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_25/TanhTanhsimple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_25_matmul_readvariableop_resource2simple_rnn_cell_25_biasadd_readvariableop_resource3simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4510592*
condR
while_cond_4510591*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_25/BiasAdd/ReadVariableOp)^simple_rnn_cell_25/MatMul/ReadVariableOp+^simple_rnn_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_25/BiasAdd/ReadVariableOp)simple_rnn_cell_25/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_25/MatMul/ReadVariableOp(simple_rnn_cell_25/MatMul/ReadVariableOp2X
*simple_rnn_cell_25/MatMul_1/ReadVariableOp*simple_rnn_cell_25/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ÿ@
Ë
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4507903

inputsC
1simple_rnn_cell_26_matmul_readvariableop_resource:4@@
2simple_rnn_cell_26_biasadd_readvariableop_resource:@E
3simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_26/BiasAdd/ReadVariableOp¢(simple_rnn_cell_26/MatMul/ReadVariableOp¢*simple_rnn_cell_26/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿå
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_26/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_26/BiasAddBiasAdd#simple_rnn_cell_26/MatMul:product:01simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_26/MatMul_1MatMulzeros:output:02simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_26/addAddV2#simple_rnn_cell_26/BiasAdd:output:0%simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_26/TanhTanhsimple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_26_matmul_readvariableop_resource2simple_rnn_cell_26_biasadd_readvariableop_resource3simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4507836*
condR
while_cond_4507835*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_26/BiasAdd/ReadVariableOp)^simple_rnn_cell_26/MatMul/ReadVariableOp+^simple_rnn_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_26/BiasAdd/ReadVariableOp)simple_rnn_cell_26/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_26/MatMul/ReadVariableOp(simple_rnn_cell_26/MatMul/ReadVariableOp2X
*simple_rnn_cell_26/MatMul_1/ReadVariableOp*simple_rnn_cell_26/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÝB
è
(backward_simple_rnn_8_while_body_4509646H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2G
Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0
backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_8_while_identity*
&backward_simple_rnn_8_while_identity_1*
&backward_simple_rnn_8_while_identity_2*
&backward_simple_rnn_8_while_identity_3*
&backward_simple_rnn_8_while_identity_4E
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp¢Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
?backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_8_while_placeholderVbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ô
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_8/while/simple_rnn_cell_26/MatMulMatMulFbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAddBiasAdd?backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul:product:0Mbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1MatMul)backward_simple_rnn_8_while_placeholder_2Nbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_8/while/simple_rnn_cell_26/addAddV2?backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd:output:0Abackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_8/while/simple_rnn_cell_26/TanhTanh6backward_simple_rnn_8/while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_8_while_placeholder_1Obackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_8/while/addAddV2'backward_simple_rnn_8_while_placeholder*backward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_8/while/add_1AddV2Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counter,backward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_8/while/IdentityIdentity%backward_simple_rnn_8/while/add_1:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_8/while/Identity_1IdentityJbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_8/while/Identity_2Identity#backward_simple_rnn_8/while/add:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_8/while/Identity_3IdentityPbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_8/while/Identity_4Identity7backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0!^backward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_8/while/NoOpNoOpF^backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpE^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpG^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0"U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0"Y
&backward_simple_rnn_8_while_identity_1/backward_simple_rnn_8/while/Identity_1:output:0"Y
&backward_simple_rnn_8_while_identity_2/backward_simple_rnn_8/while/Identity_2:output:0"Y
&backward_simple_rnn_8_while_identity_3/backward_simple_rnn_8/while/Identity_3:output:0"Y
&backward_simple_rnn_8_while_identity_4/backward_simple_rnn_8/while/Identity_4:output:0"¢
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resourcePbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resourceQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resourceObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0"
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpEbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpDbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp2
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpFbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ÔB
è
(backward_simple_rnn_8_while_body_4510086H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2G
Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0
backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_8_while_identity*
&backward_simple_rnn_8_while_identity_1*
&backward_simple_rnn_8_while_identity_2*
&backward_simple_rnn_8_while_identity_3*
&backward_simple_rnn_8_while_identity_4E
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp¢Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_8_while_placeholderVbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_8/while/simple_rnn_cell_26/MatMulMatMulFbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAddBiasAdd?backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul:product:0Mbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1MatMul)backward_simple_rnn_8_while_placeholder_2Nbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_8/while/simple_rnn_cell_26/addAddV2?backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd:output:0Abackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_8/while/simple_rnn_cell_26/TanhTanh6backward_simple_rnn_8/while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_8_while_placeholder_1Obackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_8/while/addAddV2'backward_simple_rnn_8_while_placeholder*backward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_8/while/add_1AddV2Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counter,backward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_8/while/IdentityIdentity%backward_simple_rnn_8/while/add_1:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_8/while/Identity_1IdentityJbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_8/while/Identity_2Identity#backward_simple_rnn_8/while/add:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_8/while/Identity_3IdentityPbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_8/while/Identity_4Identity7backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0!^backward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_8/while/NoOpNoOpF^backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpE^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpG^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0"U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0"Y
&backward_simple_rnn_8_while_identity_1/backward_simple_rnn_8/while/Identity_1:output:0"Y
&backward_simple_rnn_8_while_identity_2/backward_simple_rnn_8/while/Identity_2:output:0"Y
&backward_simple_rnn_8_while_identity_3/backward_simple_rnn_8/while/Identity_3:output:0"Y
&backward_simple_rnn_8_while_identity_4/backward_simple_rnn_8/while/Identity_4:output:0"¢
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resourcePbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resourceQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resourceObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0"
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpEbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpDbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp2
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpFbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ÛA
É
'forward_simple_rnn_8_while_body_4508581F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2E
Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0
}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_8_while_identity)
%forward_simple_rnn_8_while_identity_1)
%forward_simple_rnn_8_while_identity_2)
%forward_simple_rnn_8_while_identity_3)
%forward_simple_rnn_8_while_identity_4C
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp¢Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
Lforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
>forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_8_while_placeholderUforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ò
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_8/while/simple_rnn_cell_25/MatMulMatMulEforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAddBiasAdd>forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul:product:0Lforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1MatMul(forward_simple_rnn_8_while_placeholder_2Mforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_8/while/simple_rnn_cell_25/addAddV2>forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd:output:0@forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_8/while/simple_rnn_cell_25/TanhTanh5forward_simple_rnn_8/while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_8_while_placeholder_1Nforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_8/while/addAddV2&forward_simple_rnn_8_while_placeholder)forward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_8/while/add_1AddV2Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counter+forward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_8/while/IdentityIdentity$forward_simple_rnn_8/while/add_1:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_8/while/Identity_1IdentityHforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_8/while/Identity_2Identity"forward_simple_rnn_8/while/add:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_8/while/Identity_3IdentityOforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_8/while/Identity_4Identity6forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0 ^forward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_8/while/NoOpNoOpE^forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpD^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpF^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0"S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0"W
%forward_simple_rnn_8_while_identity_1.forward_simple_rnn_8/while/Identity_1:output:0"W
%forward_simple_rnn_8_while_identity_2.forward_simple_rnn_8/while/Identity_2:output:0"W
%forward_simple_rnn_8_while_identity_3.forward_simple_rnn_8/while/Identity_3:output:0"W
%forward_simple_rnn_8_while_identity_4.forward_simple_rnn_8/while/Identity_4:output:0" 
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resourceOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resourcePforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resourceNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpDforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpCforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp2
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpEforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 

ì
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4511433

inputs
states_00
matmul_readvariableop_resource:4@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
½"
ß
while_body_4507134
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_25_4507156_0:4@0
"while_simple_rnn_cell_25_4507158_0:@4
"while_simple_rnn_cell_25_4507160_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_25_4507156:4@.
 while_simple_rnn_cell_25_4507158:@2
 while_simple_rnn_cell_25_4507160:@@¢0while/simple_rnn_cell_25/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_25_4507156_0"while_simple_rnn_cell_25_4507158_0"while_simple_rnn_cell_25_4507160_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4507120r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_25/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_25_4507156"while_simple_rnn_cell_25_4507156_0"F
 while_simple_rnn_cell_25_4507158"while_simple_rnn_cell_25_4507158_0"F
 while_simple_rnn_cell_25_4507160"while_simple_rnn_cell_25_4507160_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_25/StatefulPartitionedCall0while/simple_rnn_cell_25/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 

ì
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4511478

inputs
states_00
matmul_readvariableop_resource:4@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
Ù@
Í
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511147
inputs_0C
1simple_rnn_cell_26_matmul_readvariableop_resource:4@@
2simple_rnn_cell_26_biasadd_readvariableop_resource:@E
3simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_26/BiasAdd/ReadVariableOp¢(simple_rnn_cell_26/MatMul/ReadVariableOp¢*simple_rnn_cell_26/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
(simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_26/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_26/BiasAddBiasAdd#simple_rnn_cell_26/MatMul:product:01simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_26/MatMul_1MatMulzeros:output:02simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_26/addAddV2#simple_rnn_cell_26/BiasAdd:output:0%simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_26/TanhTanhsimple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_26_matmul_readvariableop_resource2simple_rnn_cell_26_biasadd_readvariableop_resource3simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4511080*
condR
while_cond_4511079*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_26/BiasAdd/ReadVariableOp)^simple_rnn_cell_26/MatMul/ReadVariableOp+^simple_rnn_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_26/BiasAdd/ReadVariableOp)simple_rnn_cell_26/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_26/MatMul/ReadVariableOp(simple_rnn_cell_26/MatMul/ReadVariableOp2X
*simple_rnn_cell_26/MatMul_1/ReadVariableOp*simple_rnn_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
½"
ß
while_body_4507432
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_26_4507454_0:4@0
"while_simple_rnn_cell_26_4507456_0:@4
"while_simple_rnn_cell_26_4507458_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_26_4507454:4@.
 while_simple_rnn_cell_26_4507456:@2
 while_simple_rnn_cell_26_4507458:@@¢0while/simple_rnn_cell_26/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_26_4507454_0"while_simple_rnn_cell_26_4507456_0"while_simple_rnn_cell_26_4507458_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4507418r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_26/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_26_4507454"while_simple_rnn_cell_26_4507454_0"F
 while_simple_rnn_cell_26_4507456"while_simple_rnn_cell_26_4507456_0"F
 while_simple_rnn_cell_26_4507458"while_simple_rnn_cell_26_4507458_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_26/StatefulPartitionedCall0while/simple_rnn_cell_26/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 

¾
'forward_simple_rnn_8_while_cond_4509537F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2H
Dforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509537___redundant_placeholder0_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509537___redundant_placeholder1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509537___redundant_placeholder2_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509537___redundant_placeholder3'
#forward_simple_rnn_8_while_identity
¶
forward_simple_rnn_8/while/LessLess&forward_simple_rnn_8_while_placeholderDforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_8/while/IdentityIdentity#forward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
í

Fsequential_15_bidirectional_15_forward_simple_rnn_8_while_cond_4506887
sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_loop_counter
sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_maximum_iterationsI
Esequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholderK
Gsequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholder_1K
Gsequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholder_2
sequential_15_bidirectional_15_forward_simple_rnn_8_while_less_sequential_15_bidirectional_15_forward_simple_rnn_8_strided_slice_1
sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_cond_4506887___redundant_placeholder0
sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_cond_4506887___redundant_placeholder1
sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_cond_4506887___redundant_placeholder2
sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_cond_4506887___redundant_placeholder3F
Bsequential_15_bidirectional_15_forward_simple_rnn_8_while_identity
³
>sequential_15/bidirectional_15/forward_simple_rnn_8/while/LessLessEsequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholdersequential_15_bidirectional_15_forward_simple_rnn_8_while_less_sequential_15_bidirectional_15_forward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: ³
Bsequential_15/bidirectional_15/forward_simple_rnn_8/while/IdentityIdentityBsequential_15/bidirectional_15/forward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "
Bsequential_15_bidirectional_15_forward_simple_rnn_8_while_identityKsequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ù^
õ
Fsequential_15_bidirectional_15_forward_simple_rnn_8_while_body_4506888
sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_loop_counter
sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_maximum_iterationsI
Esequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholderK
Gsequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholder_1K
Gsequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholder_2
sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_strided_slice_1_0À
»sequential_15_bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0
msequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@|
nsequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@
osequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@F
Bsequential_15_bidirectional_15_forward_simple_rnn_8_while_identityH
Dsequential_15_bidirectional_15_forward_simple_rnn_8_while_identity_1H
Dsequential_15_bidirectional_15_forward_simple_rnn_8_while_identity_2H
Dsequential_15_bidirectional_15_forward_simple_rnn_8_while_identity_3H
Dsequential_15_bidirectional_15_forward_simple_rnn_8_while_identity_4
}sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_strided_slice_1¾
¹sequential_15_bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor}
ksequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource:4@z
lsequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource:@
msequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢csequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢bsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp¢dsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp¼
ksequential_15/bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   «
]sequential_15/bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem»sequential_15_bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0Esequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholdertsequential_15/bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0
bsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpmsequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0á
Ssequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMulMatMuldsequential_15/bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0jsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
csequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpnsequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ý
Tsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAddBiasAdd]sequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul:product:0ksequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOposequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0È
Usequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1MatMulGsequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholder_2lsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ë
Psequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/addAddV2]sequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd:output:0_sequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@á
Qsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/TanhTanhTsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
dsequential_15/bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Â
^sequential_15/bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemGsequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholder_1msequential_15/bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:0Usequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ
?sequential_15/bidirectional_15/forward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ø
=sequential_15/bidirectional_15/forward_simple_rnn_8/while/addAddV2Esequential_15_bidirectional_15_forward_simple_rnn_8_while_placeholderHsequential_15/bidirectional_15/forward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: 
Asequential_15/bidirectional_15/forward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¸
?sequential_15/bidirectional_15/forward_simple_rnn_8/while/add_1AddV2sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_loop_counterJsequential_15/bidirectional_15/forward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: õ
Bsequential_15/bidirectional_15/forward_simple_rnn_8/while/IdentityIdentityCsequential_15/bidirectional_15/forward_simple_rnn_8/while/add_1:z:0?^sequential_15/bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: »
Dsequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity_1Identitysequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_while_maximum_iterations?^sequential_15/bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: õ
Dsequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity_2IdentityAsequential_15/bidirectional_15/forward_simple_rnn_8/while/add:z:0?^sequential_15/bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ¢
Dsequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity_3Identitynsequential_15/bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0?^sequential_15/bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
Dsequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity_4IdentityUsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0?^sequential_15/bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²
>sequential_15/bidirectional_15/forward_simple_rnn_8/while/NoOpNoOpd^sequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpc^sequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpe^sequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Bsequential_15_bidirectional_15_forward_simple_rnn_8_while_identityKsequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity:output:0"
Dsequential_15_bidirectional_15_forward_simple_rnn_8_while_identity_1Msequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity_1:output:0"
Dsequential_15_bidirectional_15_forward_simple_rnn_8_while_identity_2Msequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity_2:output:0"
Dsequential_15_bidirectional_15_forward_simple_rnn_8_while_identity_3Msequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity_3:output:0"
Dsequential_15_bidirectional_15_forward_simple_rnn_8_while_identity_4Msequential_15/bidirectional_15/forward_simple_rnn_8/while/Identity_4:output:0"
}sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_strided_slice_1sequential_15_bidirectional_15_forward_simple_rnn_8_while_sequential_15_bidirectional_15_forward_simple_rnn_8_strided_slice_1_0"Þ
lsequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resourcensequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"à
msequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resourceosequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"Ü
ksequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resourcemsequential_15_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0"ú
¹sequential_15_bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor»sequential_15_bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_15_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2Ê
csequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpcsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2È
bsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpbsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp2Ì
dsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpdsequential_15/bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
½"
ß
while_body_4507595
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_26_4507617_0:4@0
"while_simple_rnn_cell_26_4507619_0:@4
"while_simple_rnn_cell_26_4507621_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_26_4507617:4@.
 while_simple_rnn_cell_26_4507619:@2
 while_simple_rnn_cell_26_4507621:@@¢0while/simple_rnn_cell_26/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_26_4507617_0"while_simple_rnn_cell_26_4507619_0"while_simple_rnn_cell_26_4507621_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4507540r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_26/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_26_4507617"while_simple_rnn_cell_26_4507617_0"F
 while_simple_rnn_cell_26_4507619"while_simple_rnn_cell_26_4507619_0"F
 while_simple_rnn_cell_26_4507621"while_simple_rnn_cell_26_4507621_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_26/StatefulPartitionedCall0while/simple_rnn_cell_26/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ß
¯
while_cond_4510481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4510481___redundant_placeholder05
1while_while_cond_4510481___redundant_placeholder15
1while_while_cond_4510481___redundant_placeholder25
1while_while_cond_4510481___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:

¾
'forward_simple_rnn_8_while_cond_4508580F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2H
Dforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4508580___redundant_placeholder0_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4508580___redundant_placeholder1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4508580___redundant_placeholder2_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4508580___redundant_placeholder3'
#forward_simple_rnn_8_while_identity
¶
forward_simple_rnn_8/while/LessLess&forward_simple_rnn_8_while_placeholderDforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_8/while/IdentityIdentity#forward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ÔB
è
(backward_simple_rnn_8_while_body_4508689H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2G
Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0
backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_8_while_identity*
&backward_simple_rnn_8_while_identity_1*
&backward_simple_rnn_8_while_identity_2*
&backward_simple_rnn_8_while_identity_3*
&backward_simple_rnn_8_while_identity_4E
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp¢Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_8_while_placeholderVbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_8/while/simple_rnn_cell_26/MatMulMatMulFbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAddBiasAdd?backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul:product:0Mbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1MatMul)backward_simple_rnn_8_while_placeholder_2Nbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_8/while/simple_rnn_cell_26/addAddV2?backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd:output:0Abackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_8/while/simple_rnn_cell_26/TanhTanh6backward_simple_rnn_8/while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_8_while_placeholder_1Obackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_8/while/addAddV2'backward_simple_rnn_8_while_placeholder*backward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_8/while/add_1AddV2Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counter,backward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_8/while/IdentityIdentity%backward_simple_rnn_8/while/add_1:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_8/while/Identity_1IdentityJbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_8/while/Identity_2Identity#backward_simple_rnn_8/while/add:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_8/while/Identity_3IdentityPbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_8/while/Identity_4Identity7backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0!^backward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_8/while/NoOpNoOpF^backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpE^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpG^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0"U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0"Y
&backward_simple_rnn_8_while_identity_1/backward_simple_rnn_8/while/Identity_1:output:0"Y
&backward_simple_rnn_8_while_identity_2/backward_simple_rnn_8/while/Identity_2:output:0"Y
&backward_simple_rnn_8_while_identity_3/backward_simple_rnn_8/while/Identity_3:output:0"Y
&backward_simple_rnn_8_while_identity_4/backward_simple_rnn_8/while/Identity_4:output:0"¢
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resourcePbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resourceQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resourceObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0"
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpEbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpDbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp2
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpFbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ü-
Ò
while_body_4507717
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_25_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_25/MatMul/ReadVariableOp¢0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_25/BiasAddBiasAdd)while/simple_rnn_cell_25/MatMul:product:07while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_25/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_25/addAddV2)while/simple_rnn_cell_25/BiasAdd:output:0+while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_25/TanhTanh while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_25/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_25/MatMul/ReadVariableOp1^while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_25_matmul_readvariableop_resource9while_simple_rnn_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_25/MatMul/ReadVariableOp.while/simple_rnn_cell_25/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ç
	
8bidirectional_15_forward_simple_rnn_8_while_cond_4509015h
dbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_loop_countern
jbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_maximum_iterations;
7bidirectional_15_forward_simple_rnn_8_while_placeholder=
9bidirectional_15_forward_simple_rnn_8_while_placeholder_1=
9bidirectional_15_forward_simple_rnn_8_while_placeholder_2j
fbidirectional_15_forward_simple_rnn_8_while_less_bidirectional_15_forward_simple_rnn_8_strided_slice_1
}bidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_cond_4509015___redundant_placeholder0
}bidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_cond_4509015___redundant_placeholder1
}bidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_cond_4509015___redundant_placeholder2
}bidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_cond_4509015___redundant_placeholder38
4bidirectional_15_forward_simple_rnn_8_while_identity
ú
0bidirectional_15/forward_simple_rnn_8/while/LessLess7bidirectional_15_forward_simple_rnn_8_while_placeholderfbidirectional_15_forward_simple_rnn_8_while_less_bidirectional_15_forward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: 
4bidirectional_15/forward_simple_rnn_8/while/IdentityIdentity4bidirectional_15/forward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "u
4bidirectional_15_forward_simple_rnn_8_while_identity=bidirectional_15/forward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
þ6
¬
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4507496

inputs,
simple_rnn_cell_26_4507419:4@(
simple_rnn_cell_26_4507421:@,
simple_rnn_cell_26_4507423:@@
identity¢*simple_rnn_cell_26/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskó
*simple_rnn_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_26_4507419simple_rnn_cell_26_4507421simple_rnn_cell_26_4507423*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4507418n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_26_4507419simple_rnn_cell_26_4507421simple_rnn_cell_26_4507423*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4507432*
condR
while_cond_4507431*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
NoOpNoOp+^simple_rnn_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_26/StatefulPartitionedCall*simple_rnn_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ß
¯
while_cond_4507294
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4507294___redundant_placeholder05
1while_while_cond_4507294___redundant_placeholder15
1while_while_cond_4507294___redundant_placeholder25
1while_while_cond_4507294___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ó-
Ò
while_body_4510482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_25_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_25/MatMul/ReadVariableOp¢0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_25/BiasAddBiasAdd)while/simple_rnn_cell_25/MatMul:product:07while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_25/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_25/addAddV2)while/simple_rnn_cell_25/BiasAdd:output:0+while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_25/TanhTanh while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_25/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_25/MatMul/ReadVariableOp1^while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_25_matmul_readvariableop_resource9while_simple_rnn_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_25/MatMul/ReadVariableOp.while/simple_rnn_cell_25/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ü-
Ò
while_body_4510812
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_25_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_25/MatMul/ReadVariableOp¢0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_25/BiasAddBiasAdd)while/simple_rnn_cell_25/MatMul:product:07while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_25/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_25/addAddV2)while/simple_rnn_cell_25/BiasAdd:output:0+while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_25/TanhTanh while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_25/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_25/MatMul/ReadVariableOp1^while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_25_matmul_readvariableop_resource9while_simple_rnn_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_25/MatMul/ReadVariableOp.while/simple_rnn_cell_25/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
À

Ü
4__inference_simple_rnn_cell_25_layer_call_fn_4511385

inputs
states_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4507120o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
²
Ñ
(backward_simple_rnn_8_while_cond_4509865H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2J
Fbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4509865___redundant_placeholder0a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4509865___redundant_placeholder1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4509865___redundant_placeholder2a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4509865___redundant_placeholder3(
$backward_simple_rnn_8_while_identity
º
 backward_simple_rnn_8/while/LessLess'backward_simple_rnn_8_while_placeholderFbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_8/while/IdentityIdentity$backward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ÑR
è
9bidirectional_15_backward_simple_rnn_8_while_body_4509124j
fbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_loop_counterp
lbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_maximum_iterations<
8bidirectional_15_backward_simple_rnn_8_while_placeholder>
:bidirectional_15_backward_simple_rnn_8_while_placeholder_1>
:bidirectional_15_backward_simple_rnn_8_while_placeholder_2i
ebidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_strided_slice_1_0¦
¡bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0r
`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@o
abidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@t
bbidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@9
5bidirectional_15_backward_simple_rnn_8_while_identity;
7bidirectional_15_backward_simple_rnn_8_while_identity_1;
7bidirectional_15_backward_simple_rnn_8_while_identity_2;
7bidirectional_15_backward_simple_rnn_8_while_identity_3;
7bidirectional_15_backward_simple_rnn_8_while_identity_4g
cbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_strided_slice_1¤
bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorp
^bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource:4@m
_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource:@r
`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢Vbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Ubidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp¢Wbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp¯
^bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ê
Pbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¡bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_08bidirectional_15_backward_simple_rnn_8_while_placeholdergbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ö
Ubidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0º
Fbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMulMatMulWbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0]bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
Vbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpabidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¶
Gbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAddBiasAddPbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul:product:0^bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
Wbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpbbidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¡
Hbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1MatMul:bidirectional_15_backward_simple_rnn_8_while_placeholder_2_bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Cbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/addAddV2Pbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd:output:0Rbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
Dbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/TanhTanhGbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Wbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Qbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem:bidirectional_15_backward_simple_rnn_8_while_placeholder_1`bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:0Hbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒt
2bidirectional_15/backward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ñ
0bidirectional_15/backward_simple_rnn_8/while/addAddV28bidirectional_15_backward_simple_rnn_8_while_placeholder;bidirectional_15/backward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: v
4bidirectional_15/backward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
2bidirectional_15/backward_simple_rnn_8/while/add_1AddV2fbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_loop_counter=bidirectional_15/backward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: Î
5bidirectional_15/backward_simple_rnn_8/while/IdentityIdentity6bidirectional_15/backward_simple_rnn_8/while/add_1:z:02^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
7bidirectional_15/backward_simple_rnn_8/while/Identity_1Identitylbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_maximum_iterations2^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Î
7bidirectional_15/backward_simple_rnn_8/while/Identity_2Identity4bidirectional_15/backward_simple_rnn_8/while/add:z:02^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: û
7bidirectional_15/backward_simple_rnn_8/while/Identity_3Identityabidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ó
7bidirectional_15/backward_simple_rnn_8/while/Identity_4IdentityHbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:02^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@þ
1bidirectional_15/backward_simple_rnn_8/while/NoOpNoOpW^bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpV^bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpX^bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ì
cbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_strided_slice_1ebidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_strided_slice_1_0"w
5bidirectional_15_backward_simple_rnn_8_while_identity>bidirectional_15/backward_simple_rnn_8/while/Identity:output:0"{
7bidirectional_15_backward_simple_rnn_8_while_identity_1@bidirectional_15/backward_simple_rnn_8/while/Identity_1:output:0"{
7bidirectional_15_backward_simple_rnn_8_while_identity_2@bidirectional_15/backward_simple_rnn_8/while/Identity_2:output:0"{
7bidirectional_15_backward_simple_rnn_8_while_identity_3@bidirectional_15/backward_simple_rnn_8/while/Identity_3:output:0"{
7bidirectional_15_backward_simple_rnn_8_while_identity_4@bidirectional_15/backward_simple_rnn_8/while/Identity_4:output:0"Ä
_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resourceabidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"Æ
`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resourcebbidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0"Â
^bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0"Æ
bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor¡bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2°
Vbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpVbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2®
Ubidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpUbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp2²
Wbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpWbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
	

2__inference_bidirectional_15_layer_call_fn_4509495

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508758p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ÿ
ê
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4507418

inputs

states0
matmul_readvariableop_resource:4@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
·	

2__inference_bidirectional_15_layer_call_fn_4509444
inputs_0
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4507914p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÿ
ê
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4507242

inputs

states0
matmul_readvariableop_resource:4@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
æ
æ
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508217

inputs.
forward_simple_rnn_8_4508200:4@*
forward_simple_rnn_8_4508202:@.
forward_simple_rnn_8_4508204:@@/
backward_simple_rnn_8_4508207:4@+
backward_simple_rnn_8_4508209:@/
backward_simple_rnn_8_4508211:@@
identity¢-backward_simple_rnn_8/StatefulPartitionedCall¢,forward_simple_rnn_8/StatefulPartitionedCallÆ
,forward_simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_8_4508200forward_simple_rnn_8_4508202forward_simple_rnn_8_4508204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4508186Ë
-backward_simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_8_4508207backward_simple_rnn_8_4508209backward_simple_rnn_8_4508211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4508054M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ó
concatConcatV25forward_simple_rnn_8/StatefulPartitionedCall:output:06backward_simple_rnn_8/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp.^backward_simple_rnn_8/StatefulPartitionedCall-^forward_simple_rnn_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_simple_rnn_8/StatefulPartitionedCall-backward_simple_rnn_8/StatefulPartitionedCall2\
,forward_simple_rnn_8/StatefulPartitionedCall,forward_simple_rnn_8/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·	

2__inference_bidirectional_15_layer_call_fn_4509461
inputs_0
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508217p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
´
ü
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508490

inputs*
bidirectional_15_4508459:4@&
bidirectional_15_4508461:@*
bidirectional_15_4508463:@@*
bidirectional_15_4508465:4@&
bidirectional_15_4508467:@*
bidirectional_15_4508469:@@#
dense_15_4508484:	
dense_15_4508486:
identity¢(bidirectional_15/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall
(bidirectional_15/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_15_4508459bidirectional_15_4508461bidirectional_15_4508463bidirectional_15_4508465bidirectional_15_4508467bidirectional_15_4508469*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508458¡
 dense_15/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_15/StatefulPartitionedCall:output:0dense_15_4508484dense_15_4508486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_4508483x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_15/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_15/StatefulPartitionedCall(bidirectional_15/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
?
Ê
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4507784

inputsC
1simple_rnn_cell_25_matmul_readvariableop_resource:4@@
2simple_rnn_cell_25_biasadd_readvariableop_resource:@E
3simple_rnn_cell_25_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_25/BiasAdd/ReadVariableOp¢(simple_rnn_cell_25/MatMul/ReadVariableOp¢*simple_rnn_cell_25/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿà
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_25/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_25/BiasAddBiasAdd#simple_rnn_cell_25/MatMul:product:01simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_25/MatMul_1MatMulzeros:output:02simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_25/addAddV2#simple_rnn_cell_25/BiasAdd:output:0%simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_25/TanhTanhsimple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_25_matmul_readvariableop_resource2simple_rnn_cell_25_biasadd_readvariableop_resource3simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4507717*
condR
while_cond_4507716*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_25/BiasAdd/ReadVariableOp)^simple_rnn_cell_25/MatMul/ReadVariableOp+^simple_rnn_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_25/BiasAdd/ReadVariableOp)simple_rnn_cell_25/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_25/MatMul/ReadVariableOp(simple_rnn_cell_25/MatMul/ReadVariableOp2X
*simple_rnn_cell_25/MatMul_1/ReadVariableOp*simple_rnn_cell_25/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä

J__inference_sequential_15_layer_call_and_return_conditional_losses_4508880
bidirectional_15_input*
bidirectional_15_4508861:4@&
bidirectional_15_4508863:@*
bidirectional_15_4508865:@@*
bidirectional_15_4508867:4@&
bidirectional_15_4508869:@*
bidirectional_15_4508871:@@#
dense_15_4508874:	
dense_15_4508876:
identity¢(bidirectional_15/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall
(bidirectional_15/StatefulPartitionedCallStatefulPartitionedCallbidirectional_15_inputbidirectional_15_4508861bidirectional_15_4508863bidirectional_15_4508865bidirectional_15_4508867bidirectional_15_4508869bidirectional_15_4508871*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508458¡
 dense_15/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_15/StatefulPartitionedCall:output:0dense_15_4508874dense_15_4508876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_4508483x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_15/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_15/StatefulPartitionedCall(bidirectional_15/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_15_input


Ó
/__inference_sequential_15_layer_call_fn_4508858
bidirectional_15_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508818o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_15_input
?
Ê
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4508186

inputsC
1simple_rnn_cell_25_matmul_readvariableop_resource:4@@
2simple_rnn_cell_25_biasadd_readvariableop_resource:@E
3simple_rnn_cell_25_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_25/BiasAdd/ReadVariableOp¢(simple_rnn_cell_25/MatMul/ReadVariableOp¢*simple_rnn_cell_25/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿà
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_25/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_25/BiasAddBiasAdd#simple_rnn_cell_25/MatMul:product:01simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_25/MatMul_1MatMulzeros:output:02simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_25/addAddV2#simple_rnn_cell_25/BiasAdd:output:0%simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_25/TanhTanhsimple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_25_matmul_readvariableop_resource2simple_rnn_cell_25_biasadd_readvariableop_resource3simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4508119*
condR
while_cond_4508118*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_25/BiasAdd/ReadVariableOp)^simple_rnn_cell_25/MatMul/ReadVariableOp+^simple_rnn_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_25/BiasAdd/ReadVariableOp)simple_rnn_cell_25/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_25/MatMul/ReadVariableOp(simple_rnn_cell_25/MatMul/ReadVariableOp2X
*simple_rnn_cell_25/MatMul_1/ReadVariableOp*simple_rnn_cell_25/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

Ü
4__inference_simple_rnn_cell_26_layer_call_fn_4511461

inputs
states_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4507540o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
ÿ@
Ë
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511259

inputsC
1simple_rnn_cell_26_matmul_readvariableop_resource:4@@
2simple_rnn_cell_26_biasadd_readvariableop_resource:@E
3simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_26/BiasAdd/ReadVariableOp¢(simple_rnn_cell_26/MatMul/ReadVariableOp¢*simple_rnn_cell_26/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿå
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_26/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_26/BiasAddBiasAdd#simple_rnn_cell_26/MatMul:product:01simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_26/MatMul_1MatMulzeros:output:02simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_26/addAddV2#simple_rnn_cell_26/BiasAdd:output:0%simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_26/TanhTanhsimple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_26_matmul_readvariableop_resource2simple_rnn_cell_26_biasadd_readvariableop_resource3simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4511192*
condR
while_cond_4511191*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_26/BiasAdd/ReadVariableOp)^simple_rnn_cell_26/MatMul/ReadVariableOp+^simple_rnn_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_26/BiasAdd/ReadVariableOp)simple_rnn_cell_26/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_26/MatMul/ReadVariableOp(simple_rnn_cell_26/MatMul/ReadVariableOp2X
*simple_rnn_cell_26/MatMul_1/ReadVariableOp*simple_rnn_cell_26/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø	
É
%__inference_signature_wrapper_4508931
bidirectional_15_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallbidirectional_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_4507072o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_15_input
«
Â
6__inference_forward_simple_rnn_8_layer_call_fn_4510417
inputs_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4507359o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ü-
Ò
while_body_4511304
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_26_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_26/MatMul/ReadVariableOp¢0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_26/BiasAddBiasAdd)while/simple_rnn_cell_26/MatMul:product:07while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_26/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_26/addAddV2)while/simple_rnn_cell_26/BiasAdd:output:0+while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_26/TanhTanh while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_26/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_26/MatMul/ReadVariableOp1^while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_26_matmul_readvariableop_resource9while_simple_rnn_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_26/MatMul/ReadVariableOp.while/simple_rnn_cell_26/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
?
Ê
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510879

inputsC
1simple_rnn_cell_25_matmul_readvariableop_resource:4@@
2simple_rnn_cell_25_biasadd_readvariableop_resource:@E
3simple_rnn_cell_25_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_25/BiasAdd/ReadVariableOp¢(simple_rnn_cell_25/MatMul/ReadVariableOp¢*simple_rnn_cell_25/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿà
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_25/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_25/BiasAddBiasAdd#simple_rnn_cell_25/MatMul:product:01simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_25/MatMul_1MatMulzeros:output:02simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_25/addAddV2#simple_rnn_cell_25/BiasAdd:output:0%simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_25/TanhTanhsimple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_25_matmul_readvariableop_resource2simple_rnn_cell_25_biasadd_readvariableop_resource3simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4510812*
condR
while_cond_4510811*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_25/BiasAdd/ReadVariableOp)^simple_rnn_cell_25/MatMul/ReadVariableOp+^simple_rnn_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_25/BiasAdd/ReadVariableOp)simple_rnn_cell_25/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_25/MatMul/ReadVariableOp(simple_rnn_cell_25/MatMul/ReadVariableOp2X
*simple_rnn_cell_25/MatMul_1/ReadVariableOp*simple_rnn_cell_25/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
¯
while_cond_4507594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4507594___redundant_placeholder05
1while_while_cond_4507594___redundant_placeholder15
1while_while_cond_4507594___redundant_placeholder25
1while_while_cond_4507594___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ÛA
É
'forward_simple_rnn_8_while_body_4510198F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2E
Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0
}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_8_while_identity)
%forward_simple_rnn_8_while_identity_1)
%forward_simple_rnn_8_while_identity_2)
%forward_simple_rnn_8_while_identity_3)
%forward_simple_rnn_8_while_identity_4C
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp¢Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
Lforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
>forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_8_while_placeholderUforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ò
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_8/while/simple_rnn_cell_25/MatMulMatMulEforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAddBiasAdd>forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul:product:0Lforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1MatMul(forward_simple_rnn_8_while_placeholder_2Mforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_8/while/simple_rnn_cell_25/addAddV2>forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd:output:0@forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_8/while/simple_rnn_cell_25/TanhTanh5forward_simple_rnn_8/while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_8_while_placeholder_1Nforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_8/while/addAddV2&forward_simple_rnn_8_while_placeholder)forward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_8/while/add_1AddV2Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counter+forward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_8/while/IdentityIdentity$forward_simple_rnn_8/while/add_1:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_8/while/Identity_1IdentityHforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_8/while/Identity_2Identity"forward_simple_rnn_8/while/add:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_8/while/Identity_3IdentityOforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_8/while/Identity_4Identity6forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0 ^forward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_8/while/NoOpNoOpE^forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpD^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpF^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0"S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0"W
%forward_simple_rnn_8_while_identity_1.forward_simple_rnn_8/while/Identity_1:output:0"W
%forward_simple_rnn_8_while_identity_2.forward_simple_rnn_8/while/Identity_2:output:0"W
%forward_simple_rnn_8_while_identity_3.forward_simple_rnn_8/while/Identity_3:output:0"W
%forward_simple_rnn_8_while_identity_4.forward_simple_rnn_8/while/Identity_4:output:0" 
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resourceOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resourcePforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resourceNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpDforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpCforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp2
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpEforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ß
¯
while_cond_4510967
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4510967___redundant_placeholder05
1while_while_cond_4510967___redundant_placeholder15
1while_while_cond_4510967___redundant_placeholder25
1while_while_cond_4510967___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ÝB
è
(backward_simple_rnn_8_while_body_4509866H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2G
Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0
backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_8_while_identity*
&backward_simple_rnn_8_while_identity_1*
&backward_simple_rnn_8_while_identity_2*
&backward_simple_rnn_8_while_identity_3*
&backward_simple_rnn_8_while_identity_4E
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp¢Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
?backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_8_while_placeholderVbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ô
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_8/while/simple_rnn_cell_26/MatMulMatMulFbackward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAddBiasAdd?backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul:product:0Mbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1MatMul)backward_simple_rnn_8_while_placeholder_2Nbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_8/while/simple_rnn_cell_26/addAddV2?backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd:output:0Abackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_8/while/simple_rnn_cell_26/TanhTanh6backward_simple_rnn_8/while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_8_while_placeholder_1Obackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_8/while/addAddV2'backward_simple_rnn_8_while_placeholder*backward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_8/while/add_1AddV2Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counter,backward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_8/while/IdentityIdentity%backward_simple_rnn_8/while/add_1:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_8/while/Identity_1IdentityJbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_8/while/Identity_2Identity#backward_simple_rnn_8/while/add:z:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_8/while/Identity_3IdentityPbackward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_8/while/Identity_4Identity7backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0!^backward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_8/while/NoOpNoOpF^backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpE^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpG^backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1Cbackward_simple_rnn_8_while_backward_simple_rnn_8_strided_slice_1_0"U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0"Y
&backward_simple_rnn_8_while_identity_1/backward_simple_rnn_8/while/Identity_1:output:0"Y
&backward_simple_rnn_8_while_identity_2/backward_simple_rnn_8/while/Identity_2:output:0"Y
&backward_simple_rnn_8_while_identity_3/backward_simple_rnn_8/while/Identity_3:output:0"Y
&backward_simple_rnn_8_while_identity_4/backward_simple_rnn_8/while/Identity_4:output:0"¢
Nbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resourcePbackward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resourceQbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resourceObackward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0"
}backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpEbackward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpDbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp2
Fbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpFbackward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
äA
É
'forward_simple_rnn_8_while_body_4509758F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2E
Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0
}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_8_while_identity)
%forward_simple_rnn_8_while_identity_1)
%forward_simple_rnn_8_while_identity_2)
%forward_simple_rnn_8_while_identity_3)
%forward_simple_rnn_8_while_identity_4C
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp¢Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
Lforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
>forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_8_while_placeholderUforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ò
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_8/while/simple_rnn_cell_25/MatMulMatMulEforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAddBiasAdd>forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul:product:0Lforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1MatMul(forward_simple_rnn_8_while_placeholder_2Mforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_8/while/simple_rnn_cell_25/addAddV2>forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd:output:0@forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_8/while/simple_rnn_cell_25/TanhTanh5forward_simple_rnn_8/while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_8_while_placeholder_1Nforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_8/while/addAddV2&forward_simple_rnn_8_while_placeholder)forward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_8/while/add_1AddV2Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counter+forward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_8/while/IdentityIdentity$forward_simple_rnn_8/while/add_1:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_8/while/Identity_1IdentityHforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_8/while/Identity_2Identity"forward_simple_rnn_8/while/add:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_8/while/Identity_3IdentityOforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_8/while/Identity_4Identity6forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0 ^forward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_8/while/NoOpNoOpE^forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpD^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpF^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0"S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0"W
%forward_simple_rnn_8_while_identity_1.forward_simple_rnn_8/while/Identity_1:output:0"W
%forward_simple_rnn_8_while_identity_2.forward_simple_rnn_8/while/Identity_2:output:0"W
%forward_simple_rnn_8_while_identity_3.forward_simple_rnn_8/while/Identity_3:output:0"W
%forward_simple_rnn_8_while_identity_4.forward_simple_rnn_8/while/Identity_4:output:0" 
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resourceOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resourcePforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resourceNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpDforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpCforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp2
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpEforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
§
Í
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508458

inputsX
Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp¢@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp¢backward_simple_rnn_8/while¢>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp¢?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp¢forward_simple_rnn_8/whileP
forward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_8/strided_sliceStridedSlice#forward_simple_rnn_8/Shape:output:01forward_simple_rnn_8/strided_slice/stack:output:03forward_simple_rnn_8/strided_slice/stack_1:output:03forward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_8/zeros/packedPack+forward_simple_rnn_8/strided_slice:output:0,forward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_8/zerosFill*forward_simple_rnn_8/zeros/packed:output:0)forward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_8/transpose	Transposeinputs,forward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4n
forward_simple_rnn_8/Shape_1Shape"forward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_8/strided_slice_1StridedSlice%forward_simple_rnn_8/Shape_1:output:03forward_simple_rnn_8/strided_slice_1/stack:output:05forward_simple_rnn_8/strided_slice_1/stack_1:output:05forward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_8/TensorArrayV2TensorListReserve9forward_simple_rnn_8/TensorArrayV2/element_shape:output:0-forward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
<forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_8/transpose:y:0Sforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$forward_simple_rnn_8/strided_slice_2StridedSlice"forward_simple_rnn_8/transpose:y:03forward_simple_rnn_8/strided_slice_2/stack:output:05forward_simple_rnn_8/strided_slice_2/stack_1:output:05forward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÄ
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_8/simple_rnn_cell_25/MatMulMatMul-forward_simple_rnn_8/strided_slice_2:output:0Eforward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_8/simple_rnn_cell_25/BiasAddBiasAdd8forward_simple_rnn_8/simple_rnn_cell_25/MatMul:product:0Fforward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1MatMul#forward_simple_rnn_8/zeros:output:0Gforward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_8/simple_rnn_cell_25/addAddV28forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd:output:0:forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_8/simple_rnn_cell_25/TanhTanh/forward_simple_rnn_8/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_8/TensorArrayV2_1TensorListReserve;forward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_8/whileWhile0forward_simple_rnn_8/while/loop_counter:output:06forward_simple_rnn_8/while/maximum_iterations:output:0"forward_simple_rnn_8/time:output:0-forward_simple_rnn_8/TensorArrayV2_1:handle:0#forward_simple_rnn_8/zeros:output:0-forward_simple_rnn_8/strided_slice_1:output:0Lforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resourceGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resourceHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'forward_simple_rnn_8_while_body_4508281*3
cond+R)
'forward_simple_rnn_8_while_cond_4508280*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_8/while:output:3Nforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_8/strided_slice_3StridedSlice@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_8/strided_slice_3/stack:output:05forward_simple_rnn_8/strided_slice_3/stack_1:output:05forward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_8/transpose_1	Transpose@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
backward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_8/strided_sliceStridedSlice$backward_simple_rnn_8/Shape:output:02backward_simple_rnn_8/strided_slice/stack:output:04backward_simple_rnn_8/strided_slice/stack_1:output:04backward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_8/zeros/packedPack,backward_simple_rnn_8/strided_slice:output:0-backward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_8/zerosFill+backward_simple_rnn_8/zeros/packed:output:0*backward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_simple_rnn_8/transpose	Transposeinputs-backward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
backward_simple_rnn_8/Shape_1Shape#backward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_8/strided_slice_1StridedSlice&backward_simple_rnn_8/Shape_1:output:04backward_simple_rnn_8/strided_slice_1/stack:output:06backward_simple_rnn_8/strided_slice_1/stack_1:output:06backward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_8/TensorArrayV2TensorListReserve:backward_simple_rnn_8/TensorArrayV2/element_shape:output:0.backward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_8/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¶
backward_simple_rnn_8/ReverseV2	ReverseV2#backward_simple_rnn_8/transpose:y:0-backward_simple_rnn_8/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Kbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   §
=backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_8/ReverseV2:output:0Tbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%backward_simple_rnn_8/strided_slice_2StridedSlice#backward_simple_rnn_8/transpose:y:04backward_simple_rnn_8/strided_slice_2/stack:output:06backward_simple_rnn_8/strided_slice_2/stack_1:output:06backward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_8/simple_rnn_cell_26/MatMulMatMul.backward_simple_rnn_8/strided_slice_2:output:0Fbackward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_8/simple_rnn_cell_26/BiasAddBiasAdd9backward_simple_rnn_8/simple_rnn_cell_26/MatMul:product:0Gbackward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1MatMul$backward_simple_rnn_8/zeros:output:0Hbackward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_8/simple_rnn_cell_26/addAddV29backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd:output:0;backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_8/simple_rnn_cell_26/TanhTanh0backward_simple_rnn_8/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_8/TensorArrayV2_1TensorListReserve<backward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_8/whileWhile1backward_simple_rnn_8/while/loop_counter:output:07backward_simple_rnn_8/while/maximum_iterations:output:0#backward_simple_rnn_8/time:output:0.backward_simple_rnn_8/TensorArrayV2_1:handle:0$backward_simple_rnn_8/zeros:output:0.backward_simple_rnn_8/strided_slice_1:output:0Mbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resourceHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resourceIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *4
body,R*
(backward_simple_rnn_8_while_body_4508389*4
cond,R*
(backward_simple_rnn_8_while_cond_4508388*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_8/while:output:3Obackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_8/strided_slice_3StridedSliceAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_8/strided_slice_3/stack:output:06backward_simple_rnn_8/strided_slice_3/stack_1:output:06backward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_8/transpose_1	TransposeAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_8/strided_slice_3:output:0.backward_simple_rnn_8/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?^backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpA^backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp^backward_simple_rnn_8/while?^forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>^forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp@^forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp^forward_simple_rnn_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp2
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp2:
backward_simple_rnn_8/whilebackward_simple_rnn_8/while2
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp2
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp28
forward_simple_rnn_8/whileforward_simple_rnn_8/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ß
¯
while_cond_4510811
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4510811___redundant_placeholder05
1while_while_cond_4510811___redundant_placeholder15
1while_while_cond_4510811___redundant_placeholder25
1while_while_cond_4510811___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ó-
Ò
while_body_4510968
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_26_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_26/MatMul/ReadVariableOp¢0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_26/BiasAddBiasAdd)while/simple_rnn_cell_26/MatMul:product:07while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_26/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_26/addAddV2)while/simple_rnn_cell_26/BiasAdd:output:0+while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_26/TanhTanh while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_26/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_26/MatMul/ReadVariableOp1^while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_26_matmul_readvariableop_resource9while_simple_rnn_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_26/MatMul/ReadVariableOp.while/simple_rnn_cell_26/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ß
¯
while_cond_4510701
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4510701___redundant_placeholder05
1while_while_cond_4510701___redundant_placeholder15
1while_while_cond_4510701___redundant_placeholder25
1while_while_cond_4510701___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
á

#__inference__traced_restore_4511726
file_prefix3
 assignvariableop_dense_15_kernel:	.
 assignvariableop_1_dense_15_bias:d
Rassignvariableop_2_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel:4@n
\assignvariableop_3_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel:@@^
Passignvariableop_4_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias:@e
Sassignvariableop_5_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel:4@o
]assignvariableop_6_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel:@@_
Qassignvariableop_7_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias:@&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: =
*assignvariableop_17_adam_dense_15_kernel_m:	6
(assignvariableop_18_adam_dense_15_bias_m:l
Zassignvariableop_19_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_m:4@v
dassignvariableop_20_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_m:@@f
Xassignvariableop_21_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_m:@m
[assignvariableop_22_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_m:4@w
eassignvariableop_23_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_m:@@g
Yassignvariableop_24_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_m:@=
*assignvariableop_25_adam_dense_15_kernel_v:	6
(assignvariableop_26_adam_dense_15_bias_v:l
Zassignvariableop_27_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_v:4@v
dassignvariableop_28_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_v:@@f
Xassignvariableop_29_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_v:@m
[assignvariableop_30_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_v:4@w
eassignvariableop_31_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_v:@@g
Yassignvariableop_32_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_v:@
identity_34¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¤
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ê
valueÀB½"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_2AssignVariableOpRassignvariableop_2_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_3AssignVariableOp\assignvariableop_3_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_4AssignVariableOpPassignvariableop_4_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_5AssignVariableOpSassignvariableop_5_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_6AssignVariableOp]assignvariableop_6_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_7AssignVariableOpQassignvariableop_7_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_15_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_15_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_19AssignVariableOpZassignvariableop_19_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_20AssignVariableOpdassignvariableop_20_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_21AssignVariableOpXassignvariableop_21_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_22AssignVariableOp[assignvariableop_22_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_23AssignVariableOpeassignvariableop_23_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_24AssignVariableOpYassignvariableop_24_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_15_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_15_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_27AssignVariableOpZassignvariableop_27_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_28AssignVariableOpdassignvariableop_28_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_29AssignVariableOpXassignvariableop_29_adam_bidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_30AssignVariableOp[assignvariableop_30_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_31AssignVariableOpeassignvariableop_31_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_32AssignVariableOpYassignvariableop_32_adam_bidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¥
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
­
Ã
7__inference_backward_simple_rnn_8_layer_call_fn_4510890
inputs_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4507496o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ÿ@
Ë
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4508054

inputsC
1simple_rnn_cell_26_matmul_readvariableop_resource:4@@
2simple_rnn_cell_26_biasadd_readvariableop_resource:@E
3simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_26/BiasAdd/ReadVariableOp¢(simple_rnn_cell_26/MatMul/ReadVariableOp¢*simple_rnn_cell_26/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿå
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_26/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_26/BiasAddBiasAdd#simple_rnn_cell_26/MatMul:product:01simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_26/MatMul_1MatMulzeros:output:02simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_26/addAddV2#simple_rnn_cell_26/BiasAdd:output:0%simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_26/TanhTanhsimple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_26_matmul_readvariableop_resource2simple_rnn_cell_26_biasadd_readvariableop_resource3simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4507987*
condR
while_cond_4507986*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_26/BiasAdd/ReadVariableOp)^simple_rnn_cell_26/MatMul/ReadVariableOp+^simple_rnn_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_26/BiasAdd/ReadVariableOp)simple_rnn_cell_26/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_26/MatMul/ReadVariableOp(simple_rnn_cell_26/MatMul/ReadVariableOp2X
*simple_rnn_cell_26/MatMul_1/ReadVariableOp*simple_rnn_cell_26/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
Í
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508758

inputsX
Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp¢@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp¢backward_simple_rnn_8/while¢>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp¢?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp¢forward_simple_rnn_8/whileP
forward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_8/strided_sliceStridedSlice#forward_simple_rnn_8/Shape:output:01forward_simple_rnn_8/strided_slice/stack:output:03forward_simple_rnn_8/strided_slice/stack_1:output:03forward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_8/zeros/packedPack+forward_simple_rnn_8/strided_slice:output:0,forward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_8/zerosFill*forward_simple_rnn_8/zeros/packed:output:0)forward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_8/transpose	Transposeinputs,forward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4n
forward_simple_rnn_8/Shape_1Shape"forward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_8/strided_slice_1StridedSlice%forward_simple_rnn_8/Shape_1:output:03forward_simple_rnn_8/strided_slice_1/stack:output:05forward_simple_rnn_8/strided_slice_1/stack_1:output:05forward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_8/TensorArrayV2TensorListReserve9forward_simple_rnn_8/TensorArrayV2/element_shape:output:0-forward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
<forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_8/transpose:y:0Sforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$forward_simple_rnn_8/strided_slice_2StridedSlice"forward_simple_rnn_8/transpose:y:03forward_simple_rnn_8/strided_slice_2/stack:output:05forward_simple_rnn_8/strided_slice_2/stack_1:output:05forward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÄ
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_8/simple_rnn_cell_25/MatMulMatMul-forward_simple_rnn_8/strided_slice_2:output:0Eforward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_8/simple_rnn_cell_25/BiasAddBiasAdd8forward_simple_rnn_8/simple_rnn_cell_25/MatMul:product:0Fforward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1MatMul#forward_simple_rnn_8/zeros:output:0Gforward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_8/simple_rnn_cell_25/addAddV28forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd:output:0:forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_8/simple_rnn_cell_25/TanhTanh/forward_simple_rnn_8/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_8/TensorArrayV2_1TensorListReserve;forward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_8/whileWhile0forward_simple_rnn_8/while/loop_counter:output:06forward_simple_rnn_8/while/maximum_iterations:output:0"forward_simple_rnn_8/time:output:0-forward_simple_rnn_8/TensorArrayV2_1:handle:0#forward_simple_rnn_8/zeros:output:0-forward_simple_rnn_8/strided_slice_1:output:0Lforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resourceGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resourceHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'forward_simple_rnn_8_while_body_4508581*3
cond+R)
'forward_simple_rnn_8_while_cond_4508580*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_8/while:output:3Nforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_8/strided_slice_3StridedSlice@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_8/strided_slice_3/stack:output:05forward_simple_rnn_8/strided_slice_3/stack_1:output:05forward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_8/transpose_1	Transpose@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
backward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_8/strided_sliceStridedSlice$backward_simple_rnn_8/Shape:output:02backward_simple_rnn_8/strided_slice/stack:output:04backward_simple_rnn_8/strided_slice/stack_1:output:04backward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_8/zeros/packedPack,backward_simple_rnn_8/strided_slice:output:0-backward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_8/zerosFill+backward_simple_rnn_8/zeros/packed:output:0*backward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_simple_rnn_8/transpose	Transposeinputs-backward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
backward_simple_rnn_8/Shape_1Shape#backward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_8/strided_slice_1StridedSlice&backward_simple_rnn_8/Shape_1:output:04backward_simple_rnn_8/strided_slice_1/stack:output:06backward_simple_rnn_8/strided_slice_1/stack_1:output:06backward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_8/TensorArrayV2TensorListReserve:backward_simple_rnn_8/TensorArrayV2/element_shape:output:0.backward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_8/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¶
backward_simple_rnn_8/ReverseV2	ReverseV2#backward_simple_rnn_8/transpose:y:0-backward_simple_rnn_8/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Kbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   §
=backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_8/ReverseV2:output:0Tbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%backward_simple_rnn_8/strided_slice_2StridedSlice#backward_simple_rnn_8/transpose:y:04backward_simple_rnn_8/strided_slice_2/stack:output:06backward_simple_rnn_8/strided_slice_2/stack_1:output:06backward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_8/simple_rnn_cell_26/MatMulMatMul.backward_simple_rnn_8/strided_slice_2:output:0Fbackward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_8/simple_rnn_cell_26/BiasAddBiasAdd9backward_simple_rnn_8/simple_rnn_cell_26/MatMul:product:0Gbackward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1MatMul$backward_simple_rnn_8/zeros:output:0Hbackward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_8/simple_rnn_cell_26/addAddV29backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd:output:0;backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_8/simple_rnn_cell_26/TanhTanh0backward_simple_rnn_8/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_8/TensorArrayV2_1TensorListReserve<backward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_8/whileWhile1backward_simple_rnn_8/while/loop_counter:output:07backward_simple_rnn_8/while/maximum_iterations:output:0#backward_simple_rnn_8/time:output:0.backward_simple_rnn_8/TensorArrayV2_1:handle:0$backward_simple_rnn_8/zeros:output:0.backward_simple_rnn_8/strided_slice_1:output:0Mbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resourceHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resourceIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *4
body,R*
(backward_simple_rnn_8_while_body_4508689*4
cond,R*
(backward_simple_rnn_8_while_cond_4508688*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_8/while:output:3Obackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_8/strided_slice_3StridedSliceAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_8/strided_slice_3/stack:output:06backward_simple_rnn_8/strided_slice_3/stack_1:output:06backward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_8/transpose_1	TransposeAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_8/strided_slice_3:output:0.backward_simple_rnn_8/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?^backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpA^backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp^backward_simple_rnn_8/while?^forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>^forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp@^forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp^forward_simple_rnn_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp2
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp2:
backward_simple_rnn_8/whilebackward_simple_rnn_8/while2
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp2
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp28
forward_simple_rnn_8/whileforward_simple_rnn_8/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ó-
Ò
while_body_4511080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_26_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_26/MatMul/ReadVariableOp¢0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_26/BiasAddBiasAdd)while/simple_rnn_cell_26/MatMul:product:07while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_26/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_26/addAddV2)while/simple_rnn_cell_26/BiasAdd:output:0+while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_26/TanhTanh while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_26/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_26/MatMul/ReadVariableOp1^while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_26_matmul_readvariableop_resource9while_simple_rnn_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_26/MatMul/ReadVariableOp.while/simple_rnn_cell_26/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ü-
Ò
while_body_4507987
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_26_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_26/MatMul/ReadVariableOp¢0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_26/BiasAddBiasAdd)while/simple_rnn_cell_26/MatMul:product:07while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_26/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_26/addAddV2)while/simple_rnn_cell_26/BiasAdd:output:0+while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_26/TanhTanh while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_26/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_26/MatMul/ReadVariableOp1^while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_26_biasadd_readvariableop_resource:while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_26_matmul_1_readvariableop_resource;while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_26_matmul_readvariableop_resource9while_simple_rnn_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_26/MatMul/ReadVariableOp.while/simple_rnn_cell_26/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp0while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ú>
Ì
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510549
inputs_0C
1simple_rnn_cell_25_matmul_readvariableop_resource:4@@
2simple_rnn_cell_25_biasadd_readvariableop_resource:@E
3simple_rnn_cell_25_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_25/BiasAdd/ReadVariableOp¢(simple_rnn_cell_25/MatMul/ReadVariableOp¢*simple_rnn_cell_25/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
(simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_25/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_25/BiasAddBiasAdd#simple_rnn_cell_25/MatMul:product:01simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_25/MatMul_1MatMulzeros:output:02simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_25/addAddV2#simple_rnn_cell_25/BiasAdd:output:0%simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_25/TanhTanhsimple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_25_matmul_readvariableop_resource2simple_rnn_cell_25_biasadd_readvariableop_resource3simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_4510482*
condR
while_cond_4510481*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_25/BiasAdd/ReadVariableOp)^simple_rnn_cell_25/MatMul/ReadVariableOp+^simple_rnn_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_25/BiasAdd/ReadVariableOp)simple_rnn_cell_25/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_25/MatMul/ReadVariableOp(simple_rnn_cell_25/MatMul/ReadVariableOp2X
*simple_rnn_cell_25/MatMul_1/ReadVariableOp*simple_rnn_cell_25/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ß
¯
while_cond_4508118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4508118___redundant_placeholder05
1while_while_cond_4508118___redundant_placeholder15
1while_while_cond_4508118___redundant_placeholder25
1while_while_cond_4508118___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
·
À
6__inference_forward_simple_rnn_8_layer_call_fn_4510428

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4507784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
äA
É
'forward_simple_rnn_8_while_body_4509538F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2E
Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0
}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_8_while_identity)
%forward_simple_rnn_8_while_identity_1)
%forward_simple_rnn_8_while_identity_2)
%forward_simple_rnn_8_while_identity_3)
%forward_simple_rnn_8_while_identity_4C
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp¢Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
Lforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
>forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_8_while_placeholderUforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ò
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_8/while/simple_rnn_cell_25/MatMulMatMulEforward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAddBiasAdd>forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul:product:0Lforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1MatMul(forward_simple_rnn_8_while_placeholder_2Mforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_8/while/simple_rnn_cell_25/addAddV2>forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd:output:0@forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_8/while/simple_rnn_cell_25/TanhTanh5forward_simple_rnn_8/while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_8_while_placeholder_1Nforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_8/while/addAddV2&forward_simple_rnn_8_while_placeholder)forward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_8/while/add_1AddV2Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counter+forward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_8/while/IdentityIdentity$forward_simple_rnn_8/while/add_1:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_8/while/Identity_1IdentityHforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_8/while/Identity_2Identity"forward_simple_rnn_8/while/add:z:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_8/while/Identity_3IdentityOforward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_8/while/Identity_4Identity6forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0 ^forward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_8/while/NoOpNoOpE^forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpD^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpF^forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1Aforward_simple_rnn_8_while_forward_simple_rnn_8_strided_slice_1_0"S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0"W
%forward_simple_rnn_8_while_identity_1.forward_simple_rnn_8/while/Identity_1:output:0"W
%forward_simple_rnn_8_while_identity_2.forward_simple_rnn_8/while/Identity_2:output:0"W
%forward_simple_rnn_8_while_identity_3.forward_simple_rnn_8/while/Identity_3:output:0"W
%forward_simple_rnn_8_while_identity_4.forward_simple_rnn_8/while/Identity_4:output:0" 
Mforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resourceOforward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resourcePforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resourceNforward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpDforward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpCforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp2
Eforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpEforward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
À

Ü
4__inference_simple_rnn_cell_25_layer_call_fn_4511399

inputs
states_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4507242o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
Ú	
Ã
/__inference_sequential_15_layer_call_fn_4508973

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508818o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ß
¯
while_cond_4507986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4507986___redundant_placeholder05
1while_while_cond_4507986___redundant_placeholder15
1while_while_cond_4507986___redundant_placeholder25
1while_while_cond_4507986___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
«
Â
6__inference_forward_simple_rnn_8_layer_call_fn_4510406
inputs_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4507198o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
´
ü
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508818

inputs*
bidirectional_15_4508799:4@&
bidirectional_15_4508801:@*
bidirectional_15_4508803:@@*
bidirectional_15_4508805:4@&
bidirectional_15_4508807:@*
bidirectional_15_4508809:@@#
dense_15_4508812:	
dense_15_4508814:
identity¢(bidirectional_15/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall
(bidirectional_15/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_15_4508799bidirectional_15_4508801bidirectional_15_4508803bidirectional_15_4508805bidirectional_15_4508807bidirectional_15_4508809*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4508758¡
 dense_15/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_15/StatefulPartitionedCall:output:0dense_15_4508812dense_15_4508814*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_4508483x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_15/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_15/StatefulPartitionedCall(bidirectional_15/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs

ì
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4511416

inputs
states_00
matmul_readvariableop_resource:4@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
¥

÷
E__inference_dense_15_layer_call_and_return_conditional_losses_4510395

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

*__inference_dense_15_layer_call_fn_4510384

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_4508483o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
'forward_simple_rnn_8_while_cond_4509977F
Bforward_simple_rnn_8_while_forward_simple_rnn_8_while_loop_counterL
Hforward_simple_rnn_8_while_forward_simple_rnn_8_while_maximum_iterations*
&forward_simple_rnn_8_while_placeholder,
(forward_simple_rnn_8_while_placeholder_1,
(forward_simple_rnn_8_while_placeholder_2H
Dforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509977___redundant_placeholder0_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509977___redundant_placeholder1_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509977___redundant_placeholder2_
[forward_simple_rnn_8_while_forward_simple_rnn_8_while_cond_4509977___redundant_placeholder3'
#forward_simple_rnn_8_while_identity
¶
forward_simple_rnn_8/while/LessLess&forward_simple_rnn_8_while_placeholderDforward_simple_rnn_8_while_less_forward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_8/while/IdentityIdentity#forward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_8_while_identity,forward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
²
Ñ
(backward_simple_rnn_8_while_cond_4510085H
Dbackward_simple_rnn_8_while_backward_simple_rnn_8_while_loop_counterN
Jbackward_simple_rnn_8_while_backward_simple_rnn_8_while_maximum_iterations+
'backward_simple_rnn_8_while_placeholder-
)backward_simple_rnn_8_while_placeholder_1-
)backward_simple_rnn_8_while_placeholder_2J
Fbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4510085___redundant_placeholder0a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4510085___redundant_placeholder1a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4510085___redundant_placeholder2a
]backward_simple_rnn_8_while_backward_simple_rnn_8_while_cond_4510085___redundant_placeholder3(
$backward_simple_rnn_8_while_identity
º
 backward_simple_rnn_8/while/LessLess'backward_simple_rnn_8_while_placeholderFbackward_simple_rnn_8_while_less_backward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_8/while/IdentityIdentity$backward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_8_while_identity-backward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:

	
9bidirectional_15_backward_simple_rnn_8_while_cond_4509123j
fbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_loop_counterp
lbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_maximum_iterations<
8bidirectional_15_backward_simple_rnn_8_while_placeholder>
:bidirectional_15_backward_simple_rnn_8_while_placeholder_1>
:bidirectional_15_backward_simple_rnn_8_while_placeholder_2l
hbidirectional_15_backward_simple_rnn_8_while_less_bidirectional_15_backward_simple_rnn_8_strided_slice_1
bidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_cond_4509123___redundant_placeholder0
bidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_cond_4509123___redundant_placeholder1
bidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_cond_4509123___redundant_placeholder2
bidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_cond_4509123___redundant_placeholder39
5bidirectional_15_backward_simple_rnn_8_while_identity
þ
1bidirectional_15/backward_simple_rnn_8/while/LessLess8bidirectional_15_backward_simple_rnn_8_while_placeholderhbidirectional_15_backward_simple_rnn_8_while_less_bidirectional_15_backward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: 
5bidirectional_15/backward_simple_rnn_8/while/IdentityIdentity5bidirectional_15/backward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "w
5bidirectional_15_backward_simple_rnn_8_while_identity>bidirectional_15/backward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
û§
Ï
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4509935
inputs_0X
Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp¢@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp¢backward_simple_rnn_8/while¢>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp¢?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp¢forward_simple_rnn_8/whileR
forward_simple_rnn_8/ShapeShapeinputs_0*
T0*
_output_shapes
:r
(forward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_8/strided_sliceStridedSlice#forward_simple_rnn_8/Shape:output:01forward_simple_rnn_8/strided_slice/stack:output:03forward_simple_rnn_8/strided_slice/stack_1:output:03forward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_8/zeros/packedPack+forward_simple_rnn_8/strided_slice:output:0,forward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_8/zerosFill*forward_simple_rnn_8/zeros/packed:output:0)forward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
forward_simple_rnn_8/transpose	Transposeinputs_0,forward_simple_rnn_8/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn
forward_simple_rnn_8/Shape_1Shape"forward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_8/strided_slice_1StridedSlice%forward_simple_rnn_8/Shape_1:output:03forward_simple_rnn_8/strided_slice_1/stack:output:05forward_simple_rnn_8/strided_slice_1/stack_1:output:05forward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_8/TensorArrayV2TensorListReserve9forward_simple_rnn_8/TensorArrayV2/element_shape:output:0-forward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
<forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_8/transpose:y:0Sforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
$forward_simple_rnn_8/strided_slice_2StridedSlice"forward_simple_rnn_8/transpose:y:03forward_simple_rnn_8/strided_slice_2/stack:output:05forward_simple_rnn_8/strided_slice_2/stack_1:output:05forward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÄ
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_8/simple_rnn_cell_25/MatMulMatMul-forward_simple_rnn_8/strided_slice_2:output:0Eforward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_8/simple_rnn_cell_25/BiasAddBiasAdd8forward_simple_rnn_8/simple_rnn_cell_25/MatMul:product:0Fforward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1MatMul#forward_simple_rnn_8/zeros:output:0Gforward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_8/simple_rnn_cell_25/addAddV28forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd:output:0:forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_8/simple_rnn_cell_25/TanhTanh/forward_simple_rnn_8/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_8/TensorArrayV2_1TensorListReserve;forward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_8/whileWhile0forward_simple_rnn_8/while/loop_counter:output:06forward_simple_rnn_8/while/maximum_iterations:output:0"forward_simple_rnn_8/time:output:0-forward_simple_rnn_8/TensorArrayV2_1:handle:0#forward_simple_rnn_8/zeros:output:0-forward_simple_rnn_8/strided_slice_1:output:0Lforward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resourceGforward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resourceHforward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'forward_simple_rnn_8_while_body_4509758*3
cond+R)
'forward_simple_rnn_8_while_cond_4509757*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_8/while:output:3Nforward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_8/strided_slice_3StridedSlice@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_8/strided_slice_3/stack:output:05forward_simple_rnn_8/strided_slice_3/stack_1:output:05forward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_8/transpose_1	Transpose@forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
backward_simple_rnn_8/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)backward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_8/strided_sliceStridedSlice$backward_simple_rnn_8/Shape:output:02backward_simple_rnn_8/strided_slice/stack:output:04backward_simple_rnn_8/strided_slice/stack_1:output:04backward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_8/zeros/packedPack,backward_simple_rnn_8/strided_slice:output:0-backward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_8/zerosFill+backward_simple_rnn_8/zeros/packed:output:0*backward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
backward_simple_rnn_8/transpose	Transposeinputs_0-backward_simple_rnn_8/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
backward_simple_rnn_8/Shape_1Shape#backward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_8/strided_slice_1StridedSlice&backward_simple_rnn_8/Shape_1:output:04backward_simple_rnn_8/strided_slice_1/stack:output:06backward_simple_rnn_8/strided_slice_1/stack_1:output:06backward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_8/TensorArrayV2TensorListReserve:backward_simple_rnn_8/TensorArrayV2/element_shape:output:0.backward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_8/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: È
backward_simple_rnn_8/ReverseV2	ReverseV2#backward_simple_rnn_8/transpose:y:0-backward_simple_rnn_8/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Kbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ§
=backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_8/ReverseV2:output:0Tbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
%backward_simple_rnn_8/strided_slice_2StridedSlice#backward_simple_rnn_8/transpose:y:04backward_simple_rnn_8/strided_slice_2/stack:output:06backward_simple_rnn_8/strided_slice_2/stack_1:output:06backward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_8/simple_rnn_cell_26/MatMulMatMul.backward_simple_rnn_8/strided_slice_2:output:0Fbackward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_8/simple_rnn_cell_26/BiasAddBiasAdd9backward_simple_rnn_8/simple_rnn_cell_26/MatMul:product:0Gbackward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1MatMul$backward_simple_rnn_8/zeros:output:0Hbackward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_8/simple_rnn_cell_26/addAddV29backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd:output:0;backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_8/simple_rnn_cell_26/TanhTanh0backward_simple_rnn_8/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_8/TensorArrayV2_1TensorListReserve<backward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_8/whileWhile1backward_simple_rnn_8/while/loop_counter:output:07backward_simple_rnn_8/while/maximum_iterations:output:0#backward_simple_rnn_8/time:output:0.backward_simple_rnn_8/TensorArrayV2_1:handle:0$backward_simple_rnn_8/zeros:output:0.backward_simple_rnn_8/strided_slice_1:output:0Mbackward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resourceHbackward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resourceIbackward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *4
body,R*
(backward_simple_rnn_8_while_body_4509866*4
cond,R*
(backward_simple_rnn_8_while_cond_4509865*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_8/while:output:3Obackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_8/strided_slice_3StridedSliceAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_8/strided_slice_3/stack:output:06backward_simple_rnn_8/strided_slice_3/stack_1:output:06backward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_8/transpose_1	TransposeAbackward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_8/strided_slice_3:output:0.backward_simple_rnn_8/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?^backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpA^backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp^backward_simple_rnn_8/while?^forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>^forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp@^forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp^forward_simple_rnn_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp?backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp2
>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp>backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp2
@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp@backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp2:
backward_simple_rnn_8/whilebackward_simple_rnn_8/while2
>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp>forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp=forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp2
?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp?forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp28
forward_simple_rnn_8/whileforward_simple_rnn_8/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÙQ
Ê
8bidirectional_15_forward_simple_rnn_8_while_body_4509016h
dbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_loop_countern
jbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_maximum_iterations;
7bidirectional_15_forward_simple_rnn_8_while_placeholder=
9bidirectional_15_forward_simple_rnn_8_while_placeholder_1=
9bidirectional_15_forward_simple_rnn_8_while_placeholder_2g
cbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_strided_slice_1_0¤
bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0q
_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@n
`bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@s
abidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@8
4bidirectional_15_forward_simple_rnn_8_while_identity:
6bidirectional_15_forward_simple_rnn_8_while_identity_1:
6bidirectional_15_forward_simple_rnn_8_while_identity_2:
6bidirectional_15_forward_simple_rnn_8_while_identity_3:
6bidirectional_15_forward_simple_rnn_8_while_identity_4e
abidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_strided_slice_1¢
bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensoro
]bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource:4@l
^bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource:@q
_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢Ubidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Tbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp¢Vbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp®
]bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   å
Obidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_07bidirectional_15_forward_simple_rnn_8_while_placeholderfbidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ô
Tbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0·
Ebidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMulMatMulVbidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0\bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ò
Ubidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp`bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0³
Fbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAddBiasAddObidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul:product:0]bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ø
Vbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpabidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0
Gbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1MatMul9bidirectional_15_forward_simple_rnn_8_while_placeholder_2^bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
Bbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/addAddV2Obidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd:output:0Qbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
Cbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/TanhTanhFbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Vbidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Pbidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem9bidirectional_15_forward_simple_rnn_8_while_placeholder_1_bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:0Gbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒs
1bidirectional_15/forward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Î
/bidirectional_15/forward_simple_rnn_8/while/addAddV27bidirectional_15_forward_simple_rnn_8_while_placeholder:bidirectional_15/forward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: u
3bidirectional_15/forward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ÿ
1bidirectional_15/forward_simple_rnn_8/while/add_1AddV2dbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_loop_counter<bidirectional_15/forward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: Ë
4bidirectional_15/forward_simple_rnn_8/while/IdentityIdentity5bidirectional_15/forward_simple_rnn_8/while/add_1:z:01^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
6bidirectional_15/forward_simple_rnn_8/while/Identity_1Identityjbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_while_maximum_iterations1^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Ë
6bidirectional_15/forward_simple_rnn_8/while/Identity_2Identity3bidirectional_15/forward_simple_rnn_8/while/add:z:01^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ø
6bidirectional_15/forward_simple_rnn_8/while/Identity_3Identity`bidirectional_15/forward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ð
6bidirectional_15/forward_simple_rnn_8/while/Identity_4IdentityGbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/Tanh:y:01^bidirectional_15/forward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
0bidirectional_15/forward_simple_rnn_8/while/NoOpNoOpV^bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpU^bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpW^bidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "È
abidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_strided_slice_1cbidirectional_15_forward_simple_rnn_8_while_bidirectional_15_forward_simple_rnn_8_strided_slice_1_0"u
4bidirectional_15_forward_simple_rnn_8_while_identity=bidirectional_15/forward_simple_rnn_8/while/Identity:output:0"y
6bidirectional_15_forward_simple_rnn_8_while_identity_1?bidirectional_15/forward_simple_rnn_8/while/Identity_1:output:0"y
6bidirectional_15_forward_simple_rnn_8_while_identity_2?bidirectional_15/forward_simple_rnn_8/while/Identity_2:output:0"y
6bidirectional_15_forward_simple_rnn_8_while_identity_3?bidirectional_15/forward_simple_rnn_8/while/Identity_3:output:0"y
6bidirectional_15_forward_simple_rnn_8_while_identity_4?bidirectional_15/forward_simple_rnn_8/while/Identity_4:output:0"Â
^bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource`bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"Ä
_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resourceabidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"À
]bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_bidirectional_15_forward_simple_rnn_8_while_simple_rnn_cell_25_matmul_readvariableop_resource_0"Â
bidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorbidirectional_15_forward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_forward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2®
Ubidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpUbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2¬
Tbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOpTbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul/ReadVariableOp2°
Vbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOpVbidirectional_15/forward_simple_rnn_8/while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
§Ñ
ï
J__inference_sequential_15_layer_call_and_return_conditional_losses_4509427

inputsi
Wbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource:4@f
Xbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource:@k
Ybidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@j
Xbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource:4@g
Ybidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource:@l
Zbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@:
'dense_15_matmul_readvariableop_resource:	6
(dense_15_biasadd_readvariableop_resource:
identity¢Pbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Obidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp¢Qbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp¢,bidirectional_15/backward_simple_rnn_8/while¢Obidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢Nbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp¢Pbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp¢+bidirectional_15/forward_simple_rnn_8/while¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOpa
+bidirectional_15/forward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:
9bidirectional_15/forward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;bidirectional_15/forward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;bidirectional_15/forward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3bidirectional_15/forward_simple_rnn_8/strided_sliceStridedSlice4bidirectional_15/forward_simple_rnn_8/Shape:output:0Bbidirectional_15/forward_simple_rnn_8/strided_slice/stack:output:0Dbidirectional_15/forward_simple_rnn_8/strided_slice/stack_1:output:0Dbidirectional_15/forward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4bidirectional_15/forward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@å
2bidirectional_15/forward_simple_rnn_8/zeros/packedPack<bidirectional_15/forward_simple_rnn_8/strided_slice:output:0=bidirectional_15/forward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:v
1bidirectional_15/forward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Þ
+bidirectional_15/forward_simple_rnn_8/zerosFill;bidirectional_15/forward_simple_rnn_8/zeros/packed:output:0:bidirectional_15/forward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4bidirectional_15/forward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¹
/bidirectional_15/forward_simple_rnn_8/transpose	Transposeinputs=bidirectional_15/forward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
-bidirectional_15/forward_simple_rnn_8/Shape_1Shape3bidirectional_15/forward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:
;bidirectional_15/forward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_15/forward_simple_rnn_8/strided_slice_1StridedSlice6bidirectional_15/forward_simple_rnn_8/Shape_1:output:0Dbidirectional_15/forward_simple_rnn_8/strided_slice_1/stack:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_1:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Abidirectional_15/forward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
3bidirectional_15/forward_simple_rnn_8/TensorArrayV2TensorListReserveJbidirectional_15/forward_simple_rnn_8/TensorArrayV2/element_shape:output:0>bidirectional_15/forward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¬
[bidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ò
Mbidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor3bidirectional_15/forward_simple_rnn_8/transpose:y:0dbidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;bidirectional_15/forward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:§
5bidirectional_15/forward_simple_rnn_8/strided_slice_2StridedSlice3bidirectional_15/forward_simple_rnn_8/transpose:y:0Dbidirectional_15/forward_simple_rnn_8/strided_slice_2/stack:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_1:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskæ
Nbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOpWbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
?bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMulMatMul>bidirectional_15/forward_simple_rnn_8/strided_slice_2:output:0Vbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Obidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOpXbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¡
@bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAddBiasAddIbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul:product:0Wbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
Pbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOpYbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Abidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1MatMul4bidirectional_15/forward_simple_rnn_8/zeros:output:0Xbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
<bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/addAddV2Ibidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd:output:0Kbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¹
=bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/TanhTanh@bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Cbidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Bbidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :·
5bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1TensorListReserveLbidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0Kbidirectional_15/forward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
*bidirectional_15/forward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 
>bidirectional_15/forward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿz
8bidirectional_15/forward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ë	
+bidirectional_15/forward_simple_rnn_8/whileWhileAbidirectional_15/forward_simple_rnn_8/while/loop_counter:output:0Gbidirectional_15/forward_simple_rnn_8/while/maximum_iterations:output:03bidirectional_15/forward_simple_rnn_8/time:output:0>bidirectional_15/forward_simple_rnn_8/TensorArrayV2_1:handle:04bidirectional_15/forward_simple_rnn_8/zeros:output:0>bidirectional_15/forward_simple_rnn_8/strided_slice_1:output:0]bidirectional_15/forward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Wbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_readvariableop_resourceXbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_biasadd_readvariableop_resourceYbidirectional_15_forward_simple_rnn_8_simple_rnn_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *D
body<R:
8bidirectional_15_forward_simple_rnn_8_while_body_4509243*D
cond<R:
8bidirectional_15_forward_simple_rnn_8_while_cond_4509242*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations §
Vbidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   È
Hbidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack4bidirectional_15/forward_simple_rnn_8/while:output:3_bidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
;bidirectional_15/forward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
=bidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Å
5bidirectional_15/forward_simple_rnn_8/strided_slice_3StridedSliceQbidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0Dbidirectional_15/forward_simple_rnn_8/strided_slice_3/stack:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_1:output:0Fbidirectional_15/forward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
6bidirectional_15/forward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
1bidirectional_15/forward_simple_rnn_8/transpose_1	TransposeQbidirectional_15/forward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0?bidirectional_15/forward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
,bidirectional_15/backward_simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:
:bidirectional_15/backward_simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<bidirectional_15/backward_simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<bidirectional_15/backward_simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4bidirectional_15/backward_simple_rnn_8/strided_sliceStridedSlice5bidirectional_15/backward_simple_rnn_8/Shape:output:0Cbidirectional_15/backward_simple_rnn_8/strided_slice/stack:output:0Ebidirectional_15/backward_simple_rnn_8/strided_slice/stack_1:output:0Ebidirectional_15/backward_simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5bidirectional_15/backward_simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@è
3bidirectional_15/backward_simple_rnn_8/zeros/packedPack=bidirectional_15/backward_simple_rnn_8/strided_slice:output:0>bidirectional_15/backward_simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:w
2bidirectional_15/backward_simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    á
,bidirectional_15/backward_simple_rnn_8/zerosFill<bidirectional_15/backward_simple_rnn_8/zeros/packed:output:0;bidirectional_15/backward_simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5bidirectional_15/backward_simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
0bidirectional_15/backward_simple_rnn_8/transpose	Transposeinputs>bidirectional_15/backward_simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
.bidirectional_15/backward_simple_rnn_8/Shape_1Shape4bidirectional_15/backward_simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:
<bidirectional_15/backward_simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_15/backward_simple_rnn_8/strided_slice_1StridedSlice7bidirectional_15/backward_simple_rnn_8/Shape_1:output:0Ebidirectional_15/backward_simple_rnn_8/strided_slice_1/stack:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_1:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bbidirectional_15/backward_simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
4bidirectional_15/backward_simple_rnn_8/TensorArrayV2TensorListReserveKbidirectional_15/backward_simple_rnn_8/TensorArrayV2/element_shape:output:0?bidirectional_15/backward_simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5bidirectional_15/backward_simple_rnn_8/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: é
0bidirectional_15/backward_simple_rnn_8/ReverseV2	ReverseV24bidirectional_15/backward_simple_rnn_8/transpose:y:0>bidirectional_15/backward_simple_rnn_8/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4­
\bidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ú
Nbidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor9bidirectional_15/backward_simple_rnn_8/ReverseV2:output:0ebidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<bidirectional_15/backward_simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
6bidirectional_15/backward_simple_rnn_8/strided_slice_2StridedSlice4bidirectional_15/backward_simple_rnn_8/transpose:y:0Ebidirectional_15/backward_simple_rnn_8/strided_slice_2/stack:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_1:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskè
Obidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOpXbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
@bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMulMatMul?bidirectional_15/backward_simple_rnn_8/strided_slice_2:output:0Wbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Pbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpYbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Abidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAddBiasAddJbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul:product:0Xbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ì
Qbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpZbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Bbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1MatMul5bidirectional_15/backward_simple_rnn_8/zeros:output:0Ybidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
=bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/addAddV2Jbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd:output:0Lbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
>bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/TanhTanhAbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dbidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Cbidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :º
6bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1TensorListReserveMbidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/element_shape:output:0Lbidirectional_15/backward_simple_rnn_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
+bidirectional_15/backward_simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 
?bidirectional_15/backward_simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ{
9bidirectional_15/backward_simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø	
,bidirectional_15/backward_simple_rnn_8/whileWhileBbidirectional_15/backward_simple_rnn_8/while/loop_counter:output:0Hbidirectional_15/backward_simple_rnn_8/while/maximum_iterations:output:04bidirectional_15/backward_simple_rnn_8/time:output:0?bidirectional_15/backward_simple_rnn_8/TensorArrayV2_1:handle:05bidirectional_15/backward_simple_rnn_8/zeros:output:0?bidirectional_15/backward_simple_rnn_8/strided_slice_1:output:0^bidirectional_15/backward_simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Xbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_readvariableop_resourceYbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_biasadd_readvariableop_resourceZbidirectional_15_backward_simple_rnn_8_simple_rnn_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *E
body=R;
9bidirectional_15_backward_simple_rnn_8_while_body_4509351*E
cond=R;
9bidirectional_15_backward_simple_rnn_8_while_cond_4509350*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¨
Wbidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
Ibidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack5bidirectional_15/backward_simple_rnn_8/while:output:3`bidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
<bidirectional_15/backward_simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
>bidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ê
6bidirectional_15/backward_simple_rnn_8/strided_slice_3StridedSliceRbidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0Ebidirectional_15/backward_simple_rnn_8/strided_slice_3/stack:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_1:output:0Gbidirectional_15/backward_simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
7bidirectional_15/backward_simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
2bidirectional_15/backward_simple_rnn_8/transpose_1	TransposeRbidirectional_15/backward_simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional_15/backward_simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
bidirectional_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
bidirectional_15/concatConcatV2>bidirectional_15/forward_simple_rnn_8/strided_slice_3:output:0?bidirectional_15/backward_simple_rnn_8/strided_slice_3:output:0%bidirectional_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_15/MatMulMatMul bidirectional_15/concat:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
NoOpNoOpQ^bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpP^bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpR^bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp-^bidirectional_15/backward_simple_rnn_8/whileP^bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpO^bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpQ^bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp,^bidirectional_15/forward_simple_rnn_8/while ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2¤
Pbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOpPbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/BiasAdd/ReadVariableOp2¢
Obidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOpObidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul/ReadVariableOp2¦
Qbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOpQbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/MatMul_1/ReadVariableOp2\
,bidirectional_15/backward_simple_rnn_8/while,bidirectional_15/backward_simple_rnn_8/while2¢
Obidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOpObidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/BiasAdd/ReadVariableOp2 
Nbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOpNbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul/ReadVariableOp2¤
Pbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOpPbidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/MatMul_1/ReadVariableOp2Z
+bidirectional_15/forward_simple_rnn_8/while+bidirectional_15/forward_simple_rnn_8/while2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ó-
Ò
while_body_4510592
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_25_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_25_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_25/MatMul/ReadVariableOp¢0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_25/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_25_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_25/BiasAddBiasAdd)while/simple_rnn_cell_25/MatMul:product:07while/simple_rnn_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_25/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_25/addAddV2)while/simple_rnn_cell_25/BiasAdd:output:0+while/simple_rnn_cell_25/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_25/TanhTanh while/simple_rnn_cell_25/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_25/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_25/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_25/MatMul/ReadVariableOp1^while/simple_rnn_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_25_biasadd_readvariableop_resource:while_simple_rnn_cell_25_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_25_matmul_1_readvariableop_resource;while_simple_rnn_cell_25_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_25_matmul_readvariableop_resource9while_simple_rnn_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp/while/simple_rnn_cell_25/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_25/MatMul/ReadVariableOp.while/simple_rnn_cell_25/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp0while/simple_rnn_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ß
¯
while_cond_4507716
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4507716___redundant_placeholder05
1while_while_cond_4507716___redundant_placeholder15
1while_while_cond_4507716___redundant_placeholder25
1while_while_cond_4507716___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Ú	
Ã
/__inference_sequential_15_layer_call_fn_4508952

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508490o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs

	
9bidirectional_15_backward_simple_rnn_8_while_cond_4509350j
fbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_loop_counterp
lbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_maximum_iterations<
8bidirectional_15_backward_simple_rnn_8_while_placeholder>
:bidirectional_15_backward_simple_rnn_8_while_placeholder_1>
:bidirectional_15_backward_simple_rnn_8_while_placeholder_2l
hbidirectional_15_backward_simple_rnn_8_while_less_bidirectional_15_backward_simple_rnn_8_strided_slice_1
bidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_cond_4509350___redundant_placeholder0
bidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_cond_4509350___redundant_placeholder1
bidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_cond_4509350___redundant_placeholder2
bidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_cond_4509350___redundant_placeholder39
5bidirectional_15_backward_simple_rnn_8_while_identity
þ
1bidirectional_15/backward_simple_rnn_8/while/LessLess8bidirectional_15_backward_simple_rnn_8_while_placeholderhbidirectional_15_backward_simple_rnn_8_while_less_bidirectional_15_backward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: 
5bidirectional_15/backward_simple_rnn_8/while/IdentityIdentity5bidirectional_15/backward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "w
5bidirectional_15_backward_simple_rnn_8_while_identity>bidirectional_15/backward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:

¬
Gsequential_15_bidirectional_15_backward_simple_rnn_8_while_cond_4506995
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_loop_counter
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_maximum_iterationsJ
Fsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholderL
Hsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholder_1L
Hsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholder_2
sequential_15_bidirectional_15_backward_simple_rnn_8_while_less_sequential_15_bidirectional_15_backward_simple_rnn_8_strided_slice_1 
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_cond_4506995___redundant_placeholder0 
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_cond_4506995___redundant_placeholder1 
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_cond_4506995___redundant_placeholder2 
sequential_15_bidirectional_15_backward_simple_rnn_8_while_sequential_15_bidirectional_15_backward_simple_rnn_8_while_cond_4506995___redundant_placeholder3G
Csequential_15_bidirectional_15_backward_simple_rnn_8_while_identity
·
?sequential_15/bidirectional_15/backward_simple_rnn_8/while/LessLessFsequential_15_bidirectional_15_backward_simple_rnn_8_while_placeholdersequential_15_bidirectional_15_backward_simple_rnn_8_while_less_sequential_15_bidirectional_15_backward_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: µ
Csequential_15/bidirectional_15/backward_simple_rnn_8/while/IdentityIdentityCsequential_15/bidirectional_15/backward_simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "
Csequential_15_bidirectional_15_backward_simple_rnn_8_while_identityLsequential_15/bidirectional_15/backward_simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ÿ
ê
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4507120

inputs

states0
matmul_readvariableop_resource:4@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
ß
¯
while_cond_4507431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4507431___redundant_placeholder05
1while_while_cond_4507431___redundant_placeholder15
1while_while_cond_4507431___redundant_placeholder25
1while_while_cond_4507431___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ÑR
è
9bidirectional_15_backward_simple_rnn_8_while_body_4509351j
fbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_loop_counterp
lbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_maximum_iterations<
8bidirectional_15_backward_simple_rnn_8_while_placeholder>
:bidirectional_15_backward_simple_rnn_8_while_placeholder_1>
:bidirectional_15_backward_simple_rnn_8_while_placeholder_2i
ebidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_strided_slice_1_0¦
¡bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0r
`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0:4@o
abidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0:@t
bbidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0:@@9
5bidirectional_15_backward_simple_rnn_8_while_identity;
7bidirectional_15_backward_simple_rnn_8_while_identity_1;
7bidirectional_15_backward_simple_rnn_8_while_identity_2;
7bidirectional_15_backward_simple_rnn_8_while_identity_3;
7bidirectional_15_backward_simple_rnn_8_while_identity_4g
cbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_strided_slice_1¤
bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorp
^bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource:4@m
_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource:@r
`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource:@@¢Vbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp¢Ubidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp¢Wbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp¯
^bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ê
Pbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¡bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_08bidirectional_15_backward_simple_rnn_8_while_placeholdergbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ö
Ubidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpReadVariableOp`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0º
Fbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMulMatMulWbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0]bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
Vbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpReadVariableOpabidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¶
Gbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAddBiasAddPbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul:product:0^bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
Wbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpReadVariableOpbbidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¡
Hbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1MatMul:bidirectional_15_backward_simple_rnn_8_while_placeholder_2_bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Cbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/addAddV2Pbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd:output:0Rbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
Dbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/TanhTanhGbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Wbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Qbidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem:bidirectional_15_backward_simple_rnn_8_while_placeholder_1`bidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem/index:output:0Hbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒt
2bidirectional_15/backward_simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ñ
0bidirectional_15/backward_simple_rnn_8/while/addAddV28bidirectional_15_backward_simple_rnn_8_while_placeholder;bidirectional_15/backward_simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: v
4bidirectional_15/backward_simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
2bidirectional_15/backward_simple_rnn_8/while/add_1AddV2fbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_loop_counter=bidirectional_15/backward_simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: Î
5bidirectional_15/backward_simple_rnn_8/while/IdentityIdentity6bidirectional_15/backward_simple_rnn_8/while/add_1:z:02^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: 
7bidirectional_15/backward_simple_rnn_8/while/Identity_1Identitylbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_while_maximum_iterations2^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Î
7bidirectional_15/backward_simple_rnn_8/while/Identity_2Identity4bidirectional_15/backward_simple_rnn_8/while/add:z:02^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: û
7bidirectional_15/backward_simple_rnn_8/while/Identity_3Identityabidirectional_15/backward_simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ó
7bidirectional_15/backward_simple_rnn_8/while/Identity_4IdentityHbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/Tanh:y:02^bidirectional_15/backward_simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@þ
1bidirectional_15/backward_simple_rnn_8/while/NoOpNoOpW^bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpV^bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpX^bidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ì
cbidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_strided_slice_1ebidirectional_15_backward_simple_rnn_8_while_bidirectional_15_backward_simple_rnn_8_strided_slice_1_0"w
5bidirectional_15_backward_simple_rnn_8_while_identity>bidirectional_15/backward_simple_rnn_8/while/Identity:output:0"{
7bidirectional_15_backward_simple_rnn_8_while_identity_1@bidirectional_15/backward_simple_rnn_8/while/Identity_1:output:0"{
7bidirectional_15_backward_simple_rnn_8_while_identity_2@bidirectional_15/backward_simple_rnn_8/while/Identity_2:output:0"{
7bidirectional_15_backward_simple_rnn_8_while_identity_3@bidirectional_15/backward_simple_rnn_8/while/Identity_3:output:0"{
7bidirectional_15_backward_simple_rnn_8_while_identity_4@bidirectional_15/backward_simple_rnn_8/while/Identity_4:output:0"Ä
_bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resourceabidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_biasadd_readvariableop_resource_0"Æ
`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resourcebbidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_1_readvariableop_resource_0"Â
^bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource`bidirectional_15_backward_simple_rnn_8_while_simple_rnn_cell_26_matmul_readvariableop_resource_0"Æ
bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor¡bidirectional_15_backward_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_bidirectional_15_backward_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2°
Vbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOpVbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/BiasAdd/ReadVariableOp2®
Ubidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOpUbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul/ReadVariableOp2²
Wbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOpWbidirectional_15/backward_simple_rnn_8/while/simple_rnn_cell_26/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ß
¯
while_cond_4507835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4507835___redundant_placeholder05
1while_while_cond_4507835___redundant_placeholder15
1while_while_cond_4507835___redundant_placeholder25
1while_while_cond_4507835___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ÿ
ê
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4507540

inputs

states0
matmul_readvariableop_resource:4@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ4:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Í
serving_default¹
]
bidirectional_15_inputC
(serving_default_bidirectional_15_input:0ÿÿÿÿÿÿÿÿÿ4<
dense_150
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¤
´
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
Ì
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
forward_layer
backward_layer"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
X
0
1
2
3
 4
!5
6
7"
trackable_list_wrapper
X
0
1
2
3
 4
!5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
"non_trainable_variables

#layers
$metrics
%layer_regularization_losses
&layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ñ
'trace_0
(trace_1
)trace_2
*trace_32
/__inference_sequential_15_layer_call_fn_4508509
/__inference_sequential_15_layer_call_fn_4508952
/__inference_sequential_15_layer_call_fn_4508973
/__inference_sequential_15_layer_call_fn_4508858¿
¶²²
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
annotationsª *
 z'trace_0z(trace_1z)trace_2z*trace_3
Ý
+trace_0
,trace_1
-trace_2
.trace_32ò
J__inference_sequential_15_layer_call_and_return_conditional_losses_4509200
J__inference_sequential_15_layer_call_and_return_conditional_losses_4509427
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508880
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508902¿
¶²²
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
annotationsª *
 z+trace_0z,trace_1z-trace_2z.trace_3
ÜBÙ
"__inference__wrapped_model_4507072bidirectional_15_input"
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
annotationsª *
 
ó
/iter

0beta_1

1beta_2
	2decay
3learning_ratem m¡m¢m£m¤m¥ m¦!m§v¨v©vªv«v¬v­ v®!v¯"
	optimizer
,
4serving_default"
signature_map
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
£
:trace_0
;trace_1
<trace_2
=trace_32¸
2__inference_bidirectional_15_layer_call_fn_4509444
2__inference_bidirectional_15_layer_call_fn_4509461
2__inference_bidirectional_15_layer_call_fn_4509478
2__inference_bidirectional_15_layer_call_fn_4509495å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z:trace_0z;trace_1z<trace_2z=trace_3

>trace_0
?trace_1
@trace_2
Atrace_32¤
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4509715
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4509935
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4510155
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4510375å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z>trace_0z?trace_1z@trace_2zAtrace_3
Ã
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Hcell
I
state_spec"
_tf_keras_rnn_layer
Ã
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Pcell
Q
state_spec"
_tf_keras_rnn_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
Wtrace_02Ñ
*__inference_dense_15_layer_call_fn_4510384¢
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
annotationsª *
 zWtrace_0

Xtrace_02ì
E__inference_dense_15_layer_call_and_return_conditional_losses_4510395¢
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
annotationsª *
 zXtrace_0
": 	2dense_15/kernel
:2dense_15/bias
Q:O4@2?bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel
[:Y@@2Ibidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel
K:I@2=bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias
R:P4@2@bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel
\:Z@@2Jbidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel
L:J@2>bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_15_layer_call_fn_4508509bidirectional_15_input"¿
¶²²
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
annotationsª *
 
Bý
/__inference_sequential_15_layer_call_fn_4508952inputs"¿
¶²²
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
annotationsª *
 
Bý
/__inference_sequential_15_layer_call_fn_4508973inputs"¿
¶²²
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
annotationsª *
 
B
/__inference_sequential_15_layer_call_fn_4508858bidirectional_15_input"¿
¶²²
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
annotationsª *
 
B
J__inference_sequential_15_layer_call_and_return_conditional_losses_4509200inputs"¿
¶²²
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
annotationsª *
 
B
J__inference_sequential_15_layer_call_and_return_conditional_losses_4509427inputs"¿
¶²²
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
annotationsª *
 
«B¨
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508880bidirectional_15_input"¿
¶²²
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
annotationsª *
 
«B¨
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508902bidirectional_15_input"¿
¶²²
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
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÛBØ
%__inference_signature_wrapper_4508931bidirectional_15_input"
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
annotationsª *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
«B¨
2__inference_bidirectional_15_layer_call_fn_4509444inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
«B¨
2__inference_bidirectional_15_layer_call_fn_4509461inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
©B¦
2__inference_bidirectional_15_layer_call_fn_4509478inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
©B¦
2__inference_bidirectional_15_layer_call_fn_4509495inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÆBÃ
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4509715inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÆBÃ
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4509935inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÄBÁ
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4510155inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÄBÁ
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4510375inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

[states
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
¢
atrace_0
btrace_1
ctrace_2
dtrace_32·
6__inference_forward_simple_rnn_8_layer_call_fn_4510406
6__inference_forward_simple_rnn_8_layer_call_fn_4510417
6__inference_forward_simple_rnn_8_layer_call_fn_4510428
6__inference_forward_simple_rnn_8_layer_call_fn_4510439Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zatrace_0zbtrace_1zctrace_2zdtrace_3

etrace_0
ftrace_1
gtrace_2
htrace_32£
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510549
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510659
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510769
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510879Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zetrace_0zftrace_1zgtrace_2zhtrace_3
è
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
o_random_generator

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
0
 1
!2"
trackable_list_wrapper
5
0
 1
!2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

pstates
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
¦
vtrace_0
wtrace_1
xtrace_2
ytrace_32»
7__inference_backward_simple_rnn_8_layer_call_fn_4510890
7__inference_backward_simple_rnn_8_layer_call_fn_4510901
7__inference_backward_simple_rnn_8_layer_call_fn_4510912
7__inference_backward_simple_rnn_8_layer_call_fn_4510923Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zvtrace_0zwtrace_1zxtrace_2zytrace_3

ztrace_0
{trace_1
|trace_2
}trace_32§
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511035
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511147
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511259
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511371Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zztrace_0z{trace_1z|trace_2z}trace_3
í
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

kernel
 recurrent_kernel
!bias"
_tf_keras_layer
 "
trackable_list_wrapper
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
ÞBÛ
*__inference_dense_15_layer_call_fn_4510384inputs"¢
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
annotationsª *
 
ùBö
E__inference_dense_15_layer_call_and_return_conditional_losses_4510395inputs"¢
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
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
6__inference_forward_simple_rnn_8_layer_call_fn_4510406inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
6__inference_forward_simple_rnn_8_layer_call_fn_4510417inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
6__inference_forward_simple_rnn_8_layer_call_fn_4510428inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
6__inference_forward_simple_rnn_8_layer_call_fn_4510439inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¹B¶
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510549inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¹B¶
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510659inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
·B´
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510769inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
·B´
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510879inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
ç
trace_0
trace_12¬
4__inference_simple_rnn_cell_25_layer_call_fn_4511385
4__inference_simple_rnn_cell_25_layer_call_fn_4511399½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12â
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4511416
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4511433½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
7__inference_backward_simple_rnn_8_layer_call_fn_4510890inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
7__inference_backward_simple_rnn_8_layer_call_fn_4510901inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
7__inference_backward_simple_rnn_8_layer_call_fn_4510912inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
7__inference_backward_simple_rnn_8_layer_call_fn_4510923inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ºB·
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511035inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ºB·
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511147inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511259inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511371inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
 1
!2"
trackable_list_wrapper
5
0
 1
!2"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ç
trace_0
trace_12¬
4__inference_simple_rnn_cell_26_layer_call_fn_4511447
4__inference_simple_rnn_cell_26_layer_call_fn_4511461½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12â
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4511478
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4511495½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
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
B
4__inference_simple_rnn_cell_25_layer_call_fn_4511385inputsstates/0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
4__inference_simple_rnn_cell_25_layer_call_fn_4511399inputsstates/0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨B¥
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4511416inputsstates/0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨B¥
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4511433inputsstates/0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
B
4__inference_simple_rnn_cell_26_layer_call_fn_4511447inputsstates/0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
4__inference_simple_rnn_cell_26_layer_call_fn_4511461inputsstates/0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨B¥
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4511478inputsstates/0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨B¥
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4511495inputsstates/0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%	2Adam/dense_15/kernel/m
 :2Adam/dense_15/bias/m
V:T4@2FAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/m
`:^@@2PAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/m
P:N@2DAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/m
W:U4@2GAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/m
a:_@@2QAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/m
Q:O@2EAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/m
':%	2Adam/dense_15/kernel/v
 :2Adam/dense_15/bias/v
V:T4@2FAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/kernel/v
`:^@@2PAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/recurrent_kernel/v
P:N@2DAdam/bidirectional_15/forward_simple_rnn_8/simple_rnn_cell_25/bias/v
W:U4@2GAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/kernel/v
a:_@@2QAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/recurrent_kernel/v
Q:O@2EAdam/bidirectional_15/backward_simple_rnn_8/simple_rnn_cell_26/bias/v«
"__inference__wrapped_model_4507072! C¢@
9¢6
41
bidirectional_15_inputÿÿÿÿÿÿÿÿÿ4
ª "3ª0
.
dense_15"
dense_15ÿÿÿÿÿÿÿÿÿÓ
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511035}! O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 Ó
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511147}! O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 Õ
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511259! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 Õ
R__inference_backward_simple_rnn_8_layer_call_and_return_conditional_losses_4511371! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 «
7__inference_backward_simple_rnn_8_layer_call_fn_4510890p! O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@«
7__inference_backward_simple_rnn_8_layer_call_fn_4510901p! O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@­
7__inference_backward_simple_rnn_8_layer_call_fn_4510912r! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@­
7__inference_backward_simple_rnn_8_layer_call_fn_4510923r! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@à
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4509715! \¢Y
R¢O
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 à
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4509935! \¢Y
R¢O
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Æ
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4510155u! C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Æ
M__inference_bidirectional_15_layer_call_and_return_conditional_losses_4510375u! C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¸
2__inference_bidirectional_15_layer_call_fn_4509444! \¢Y
R¢O
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 

 

 
ª "ÿÿÿÿÿÿÿÿÿ¸
2__inference_bidirectional_15_layer_call_fn_4509461! \¢Y
R¢O
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 

 

 
ª "ÿÿÿÿÿÿÿÿÿ
2__inference_bidirectional_15_layer_call_fn_4509478h! C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 

 

 
ª "ÿÿÿÿÿÿÿÿÿ
2__inference_bidirectional_15_layer_call_fn_4509495h! C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 

 

 
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_dense_15_layer_call_and_return_conditional_losses_4510395]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_dense_15_layer_call_fn_4510384P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÒ
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510549}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 Ò
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510659}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 Ô
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510769Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 Ô
Q__inference_forward_simple_rnn_8_layer_call_and_return_conditional_losses_4510879Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ª
6__inference_forward_simple_rnn_8_layer_call_fn_4510406pO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@ª
6__inference_forward_simple_rnn_8_layer_call_fn_4510417pO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@¬
6__inference_forward_simple_rnn_8_layer_call_fn_4510428rQ¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@¬
6__inference_forward_simple_rnn_8_layer_call_fn_4510439rQ¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@Ì
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508880~! K¢H
A¢>
41
bidirectional_15_inputÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
J__inference_sequential_15_layer_call_and_return_conditional_losses_4508902~! K¢H
A¢>
41
bidirectional_15_inputÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
J__inference_sequential_15_layer_call_and_return_conditional_losses_4509200n! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
J__inference_sequential_15_layer_call_and_return_conditional_losses_4509427n! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¤
/__inference_sequential_15_layer_call_fn_4508509q! K¢H
A¢>
41
bidirectional_15_inputÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
/__inference_sequential_15_layer_call_fn_4508858q! K¢H
A¢>
41
bidirectional_15_inputÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_15_layer_call_fn_4508952a! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_15_layer_call_fn_4508973a! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ
%__inference_signature_wrapper_4508931! ]¢Z
¢ 
SªP
N
bidirectional_15_input41
bidirectional_15_inputÿÿÿÿÿÿÿÿÿ4"3ª0
.
dense_15"
dense_15ÿÿÿÿÿÿÿÿÿ
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4511416·\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ4
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ@
$!

0/1/0ÿÿÿÿÿÿÿÿÿ@
 
O__inference_simple_rnn_cell_25_layer_call_and_return_conditional_losses_4511433·\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ4
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ@
$!

0/1/0ÿÿÿÿÿÿÿÿÿ@
 â
4__inference_simple_rnn_cell_25_layer_call_fn_4511385©\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ4
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ@
"

1/0ÿÿÿÿÿÿÿÿÿ@â
4__inference_simple_rnn_cell_25_layer_call_fn_4511399©\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ4
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ@
"

1/0ÿÿÿÿÿÿÿÿÿ@
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4511478·! \¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ4
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ@
$!

0/1/0ÿÿÿÿÿÿÿÿÿ@
 
O__inference_simple_rnn_cell_26_layer_call_and_return_conditional_losses_4511495·! \¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ4
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ@
$!

0/1/0ÿÿÿÿÿÿÿÿÿ@
 â
4__inference_simple_rnn_cell_26_layer_call_fn_4511447©! \¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ4
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ@
"

1/0ÿÿÿÿÿÿÿÿÿ@â
4__inference_simple_rnn_cell_26_layer_call_fn_4511461©! \¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ4
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ@
"

1/0ÿÿÿÿÿÿÿÿÿ@