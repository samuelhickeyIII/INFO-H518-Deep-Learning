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
EAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*V
shared_nameGEAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/v
Û
YAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/v/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/v*
_output_shapes
:@*
dtype0
þ
QAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*b
shared_nameSQAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/v
÷
eAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpQAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
ê
GAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*X
shared_nameIGAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/v
ã
[Adam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/v*
_output_shapes

:4@*
dtype0
à
DAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*U
shared_nameFDAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/v
Ù
XAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/v/Read/ReadVariableOpReadVariableOpDAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/v*
_output_shapes
:@*
dtype0
ü
PAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*a
shared_nameRPAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/v
õ
dAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpPAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
è
FAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*W
shared_nameHFAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/v
á
ZAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/v*
_output_shapes

:4@*
dtype0

Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
:*
dtype0

Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_18/kernel/v

*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes
:	*
dtype0
â
EAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*V
shared_nameGEAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/m
Û
YAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/m/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/m*
_output_shapes
:@*
dtype0
þ
QAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*b
shared_nameSQAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/m
÷
eAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpQAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
ê
GAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*X
shared_nameIGAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/m
ã
[Adam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/m*
_output_shapes

:4@*
dtype0
à
DAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*U
shared_nameFDAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/m
Ù
XAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/m/Read/ReadVariableOpReadVariableOpDAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/m*
_output_shapes
:@*
dtype0
ü
PAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*a
shared_nameRPAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/m
õ
dAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpPAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
è
FAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*W
shared_nameHFAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/m
á
ZAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/m*
_output_shapes

:4@*
dtype0

Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
:*
dtype0

Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_18/kernel/m

*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
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
>bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*O
shared_name@>bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias
Í
Rbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/Read/ReadVariableOpReadVariableOp>bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias*
_output_shapes
:@*
dtype0
ð
Jbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*[
shared_nameLJbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel
é
^bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/Read/ReadVariableOpReadVariableOpJbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel*
_output_shapes

:@@*
dtype0
Ü
@bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*Q
shared_nameB@bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel
Õ
Tbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/Read/ReadVariableOpReadVariableOp@bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel*
_output_shapes

:4@*
dtype0
Ò
=bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*N
shared_name?=bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias
Ë
Qbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/Read/ReadVariableOpReadVariableOp=bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias*
_output_shapes
:@*
dtype0
î
Ibidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*Z
shared_nameKIbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel
ç
]bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/Read/ReadVariableOpReadVariableOpIbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel*
_output_shapes

:@@*
dtype0
Ú
?bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*P
shared_nameA?bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel
Ó
Sbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/Read/ReadVariableOpReadVariableOp?bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel*
_output_shapes

:4@*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:*
dtype0
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:	*
dtype0

&serving_default_bidirectional_18_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ4

StatefulPartitionedCallStatefulPartitionedCall&serving_default_bidirectional_18_input?bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel=bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/biasIbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel@bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel>bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/biasJbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kerneldense_18/kerneldense_18/bias*
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
%__inference_signature_wrapper_4704761

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
VARIABLE_VALUEdense_18/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_18/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEIbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE@bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
£
VARIABLE_VALUEFAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
­¦
VARIABLE_VALUEPAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¡
VARIABLE_VALUEDAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
£
VARIABLE_VALUEFAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
­¦
VARIABLE_VALUEPAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¡
VARIABLE_VALUEDAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ô
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOpSbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/Read/ReadVariableOp]bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/Read/ReadVariableOpQbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/Read/ReadVariableOpTbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/Read/ReadVariableOp^bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/Read/ReadVariableOpRbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOpZAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/m/Read/ReadVariableOpdAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/m/Read/ReadVariableOpXAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/m/Read/ReadVariableOp[Adam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/m/Read/ReadVariableOpeAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/m/Read/ReadVariableOpYAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOpZAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/v/Read/ReadVariableOpdAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/v/Read/ReadVariableOpXAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/v/Read/ReadVariableOp[Adam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/v/Read/ReadVariableOpeAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/v/Read/ReadVariableOpYAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/v/Read/ReadVariableOpConst*.
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
 __inference__traced_save_4707447
»
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_18/kerneldense_18/bias?bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernelIbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel=bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias@bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernelJbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel>bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_18/kernel/mAdam/dense_18/bias/mFAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/mPAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/mDAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/mGAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/mQAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/mEAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/mAdam/dense_18/kernel/vAdam/dense_18/bias/vFAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/vPAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/vDAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/vGAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/vQAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/vEAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/v*-
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
#__inference__traced_restore_4707556ÌÍ,
·
À
6__inference_forward_simple_rnn_9_layer_call_fn_4706269

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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4704016o
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
5
«
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4703028

inputs,
simple_rnn_cell_28_4702951:4@(
simple_rnn_cell_28_4702953:@,
simple_rnn_cell_28_4702955:@@
identity¢*simple_rnn_cell_28/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_28_4702951simple_rnn_cell_28_4702953simple_rnn_cell_28_4702955*
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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4702950n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_28_4702951simple_rnn_cell_28_4702953simple_rnn_cell_28_4702955*
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
while_body_4702964*
condR
while_cond_4702963*8
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
NoOpNoOp+^simple_rnn_cell_28/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_28/StatefulPartitionedCall*simple_rnn_cell_28/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ú>
Ì
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706489
inputs_0C
1simple_rnn_cell_28_matmul_readvariableop_resource:4@@
2simple_rnn_cell_28_biasadd_readvariableop_resource:@E
3simple_rnn_cell_28_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_28/BiasAdd/ReadVariableOp¢(simple_rnn_cell_28/MatMul/ReadVariableOp¢*simple_rnn_cell_28/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_28/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_28/BiasAddBiasAdd#simple_rnn_cell_28/MatMul:product:01simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_28/MatMul_1MatMulzeros:output:02simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_28/addAddV2#simple_rnn_cell_28/BiasAdd:output:0%simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_28/TanhTanhsimple_rnn_cell_28/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_28_matmul_readvariableop_resource2simple_rnn_cell_28_biasadd_readvariableop_resource3simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
while_body_4706422*
condR
while_cond_4706421*8
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
NoOpNoOp*^simple_rnn_cell_28/BiasAdd/ReadVariableOp)^simple_rnn_cell_28/MatMul/ReadVariableOp+^simple_rnn_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_28/BiasAdd/ReadVariableOp)simple_rnn_cell_28/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_28/MatMul/ReadVariableOp(simple_rnn_cell_28/MatMul/ReadVariableOp2X
*simple_rnn_cell_28/MatMul_1/ReadVariableOp*simple_rnn_cell_28/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ß
¯
while_cond_4706421
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4706421___redundant_placeholder05
1while_while_cond_4706421___redundant_placeholder15
1while_while_cond_4706421___redundant_placeholder25
1while_while_cond_4706421___redundant_placeholder3
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
'forward_simple_rnn_9_while_cond_4705587F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2H
Dforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705587___redundant_placeholder0_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705587___redundant_placeholder1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705587___redundant_placeholder2_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705587___redundant_placeholder3'
#forward_simple_rnn_9_while_identity
¶
forward_simple_rnn_9/while/LessLess&forward_simple_rnn_9_while_placeholderDforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_9/while/IdentityIdentity#forward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0*(
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
À

Ü
4__inference_simple_rnn_cell_29_layer_call_fn_4707291

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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4703370o
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
·	

2__inference_bidirectional_18_layer_call_fn_4705291
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704047p
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704320

inputs*
bidirectional_18_4704289:4@&
bidirectional_18_4704291:@*
bidirectional_18_4704293:@@*
bidirectional_18_4704295:4@&
bidirectional_18_4704297:@*
bidirectional_18_4704299:@@#
dense_18_4704314:	
dense_18_4704316:
identity¢(bidirectional_18/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall
(bidirectional_18/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_18_4704289bidirectional_18_4704291bidirectional_18_4704293bidirectional_18_4704295bidirectional_18_4704297bidirectional_18_4704299*
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704288¡
 dense_18/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_18/StatefulPartitionedCall:output:0dense_18_4704314dense_18_4704316*
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
E__inference_dense_18_layer_call_and_return_conditional_losses_4704313x
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_18/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_18/StatefulPartitionedCall(bidirectional_18/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
?
Ê
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4704016

inputsC
1simple_rnn_cell_28_matmul_readvariableop_resource:4@@
2simple_rnn_cell_28_biasadd_readvariableop_resource:@E
3simple_rnn_cell_28_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_28/BiasAdd/ReadVariableOp¢(simple_rnn_cell_28/MatMul/ReadVariableOp¢*simple_rnn_cell_28/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_28/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_28/BiasAddBiasAdd#simple_rnn_cell_28/MatMul:product:01simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_28/MatMul_1MatMulzeros:output:02simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_28/addAddV2#simple_rnn_cell_28/BiasAdd:output:0%simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_28/TanhTanhsimple_rnn_cell_28/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_28_matmul_readvariableop_resource2simple_rnn_cell_28_biasadd_readvariableop_resource3simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
while_body_4703949*
condR
while_cond_4703948*8
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
NoOpNoOp*^simple_rnn_cell_28/BiasAdd/ReadVariableOp)^simple_rnn_cell_28/MatMul/ReadVariableOp+^simple_rnn_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_28/BiasAdd/ReadVariableOp)simple_rnn_cell_28/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_28/MatMul/ReadVariableOp(simple_rnn_cell_28/MatMul/ReadVariableOp2X
*simple_rnn_cell_28/MatMul_1/ReadVariableOp*simple_rnn_cell_28/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä

J__inference_sequential_18_layer_call_and_return_conditional_losses_4704710
bidirectional_18_input*
bidirectional_18_4704691:4@&
bidirectional_18_4704693:@*
bidirectional_18_4704695:@@*
bidirectional_18_4704697:4@&
bidirectional_18_4704699:@*
bidirectional_18_4704701:@@#
dense_18_4704704:	
dense_18_4704706:
identity¢(bidirectional_18/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall
(bidirectional_18/StatefulPartitionedCallStatefulPartitionedCallbidirectional_18_inputbidirectional_18_4704691bidirectional_18_4704693bidirectional_18_4704695bidirectional_18_4704697bidirectional_18_4704699bidirectional_18_4704701*
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704288¡
 dense_18/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_18/StatefulPartitionedCall:output:0dense_18_4704704dense_18_4704706*
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
E__inference_dense_18_layer_call_and_return_conditional_losses_4704313x
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_18/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_18/StatefulPartitionedCall(bidirectional_18/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_18_input
Ú	
Ã
/__inference_sequential_18_layer_call_fn_4704782

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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704320o
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
ü-
Ò
while_body_4706532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_28_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_28/MatMul/ReadVariableOp¢0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_28/BiasAddBiasAdd)while/simple_rnn_cell_28/MatMul:product:07while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_28/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_28/addAddV2)while/simple_rnn_cell_28/BiasAdd:output:0+while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_28/TanhTanh while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_28/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_28/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_28/MatMul/ReadVariableOp1^while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_28_matmul_readvariableop_resource9while_simple_rnn_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_28/MatMul/ReadVariableOp.while/simple_rnn_cell_28/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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

¬
Gsequential_18_bidirectional_18_backward_simple_rnn_9_while_cond_4702825
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_loop_counter
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_maximum_iterationsJ
Fsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholderL
Hsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholder_1L
Hsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholder_2
sequential_18_bidirectional_18_backward_simple_rnn_9_while_less_sequential_18_bidirectional_18_backward_simple_rnn_9_strided_slice_1 
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_cond_4702825___redundant_placeholder0 
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_cond_4702825___redundant_placeholder1 
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_cond_4702825___redundant_placeholder2 
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_cond_4702825___redundant_placeholder3G
Csequential_18_bidirectional_18_backward_simple_rnn_9_while_identity
·
?sequential_18/bidirectional_18/backward_simple_rnn_9/while/LessLessFsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholdersequential_18_bidirectional_18_backward_simple_rnn_9_while_less_sequential_18_bidirectional_18_backward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: µ
Csequential_18/bidirectional_18/backward_simple_rnn_9/while/IdentityIdentityCsequential_18/bidirectional_18/backward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "
Csequential_18_bidirectional_18_backward_simple_rnn_9_while_identityLsequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity:output:0*(
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
while_cond_4703261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4703261___redundant_placeholder05
1while_while_cond_4703261___redundant_placeholder15
1while_while_cond_4703261___redundant_placeholder25
1while_while_cond_4703261___redundant_placeholder3
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4703370

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
ó-
Ò
while_body_4706910
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_29_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_29/MatMul/ReadVariableOp¢0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_29/BiasAddBiasAdd)while/simple_rnn_cell_29/MatMul:product:07while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_29/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_29/addAddV2)while/simple_rnn_cell_29/BiasAdd:output:0+while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_29/TanhTanh while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_29/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_29/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_29/MatMul/ReadVariableOp1^while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_29_matmul_readvariableop_resource9while_simple_rnn_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_29/MatMul/ReadVariableOp.while/simple_rnn_cell_29/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703733

inputsC
1simple_rnn_cell_29_matmul_readvariableop_resource:4@@
2simple_rnn_cell_29_biasadd_readvariableop_resource:@E
3simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_29/BiasAdd/ReadVariableOp¢(simple_rnn_cell_29/MatMul/ReadVariableOp¢*simple_rnn_cell_29/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_29/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_29/BiasAddBiasAdd#simple_rnn_cell_29/MatMul:product:01simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_29/MatMul_1MatMulzeros:output:02simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_29/addAddV2#simple_rnn_cell_29/BiasAdd:output:0%simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_29/TanhTanhsimple_rnn_cell_29/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_29_matmul_readvariableop_resource2simple_rnn_cell_29_biasadd_readvariableop_resource3simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
while_body_4703666*
condR
while_cond_4703665*8
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
NoOpNoOp*^simple_rnn_cell_29/BiasAdd/ReadVariableOp)^simple_rnn_cell_29/MatMul/ReadVariableOp+^simple_rnn_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_29/BiasAdd/ReadVariableOp)simple_rnn_cell_29/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_29/MatMul/ReadVariableOp(simple_rnn_cell_29/MatMul/ReadVariableOp2X
*simple_rnn_cell_29/MatMul_1/ReadVariableOp*simple_rnn_cell_29/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó-
Ò
while_body_4706312
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_28_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_28/MatMul/ReadVariableOp¢0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_28/BiasAddBiasAdd)while/simple_rnn_cell_28/MatMul:product:07while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_28/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_28/addAddV2)while/simple_rnn_cell_28/BiasAdd:output:0+while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_28/TanhTanh while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_28/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_28/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_28/MatMul/ReadVariableOp1^while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_28_matmul_readvariableop_resource9while_simple_rnn_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_28/MatMul/ReadVariableOp.while/simple_rnn_cell_28/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704588

inputsX
Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp¢@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp¢backward_simple_rnn_9/while¢>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp¢?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp¢forward_simple_rnn_9/whileP
forward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_9/strided_sliceStridedSlice#forward_simple_rnn_9/Shape:output:01forward_simple_rnn_9/strided_slice/stack:output:03forward_simple_rnn_9/strided_slice/stack_1:output:03forward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_9/zeros/packedPack+forward_simple_rnn_9/strided_slice:output:0,forward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_9/zerosFill*forward_simple_rnn_9/zeros/packed:output:0)forward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_9/transpose	Transposeinputs,forward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4n
forward_simple_rnn_9/Shape_1Shape"forward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_9/strided_slice_1StridedSlice%forward_simple_rnn_9/Shape_1:output:03forward_simple_rnn_9/strided_slice_1/stack:output:05forward_simple_rnn_9/strided_slice_1/stack_1:output:05forward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_9/TensorArrayV2TensorListReserve9forward_simple_rnn_9/TensorArrayV2/element_shape:output:0-forward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
<forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_9/transpose:y:0Sforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$forward_simple_rnn_9/strided_slice_2StridedSlice"forward_simple_rnn_9/transpose:y:03forward_simple_rnn_9/strided_slice_2/stack:output:05forward_simple_rnn_9/strided_slice_2/stack_1:output:05forward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÄ
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_9/simple_rnn_cell_28/MatMulMatMul-forward_simple_rnn_9/strided_slice_2:output:0Eforward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_9/simple_rnn_cell_28/BiasAddBiasAdd8forward_simple_rnn_9/simple_rnn_cell_28/MatMul:product:0Fforward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1MatMul#forward_simple_rnn_9/zeros:output:0Gforward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_9/simple_rnn_cell_28/addAddV28forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd:output:0:forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_9/simple_rnn_cell_28/TanhTanh/forward_simple_rnn_9/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_9/TensorArrayV2_1TensorListReserve;forward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_9/whileWhile0forward_simple_rnn_9/while/loop_counter:output:06forward_simple_rnn_9/while/maximum_iterations:output:0"forward_simple_rnn_9/time:output:0-forward_simple_rnn_9/TensorArrayV2_1:handle:0#forward_simple_rnn_9/zeros:output:0-forward_simple_rnn_9/strided_slice_1:output:0Lforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resourceGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resourceHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
'forward_simple_rnn_9_while_body_4704411*3
cond+R)
'forward_simple_rnn_9_while_cond_4704410*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_9/while:output:3Nforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_9/strided_slice_3StridedSlice@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_9/strided_slice_3/stack:output:05forward_simple_rnn_9/strided_slice_3/stack_1:output:05forward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_9/transpose_1	Transpose@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
backward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_9/strided_sliceStridedSlice$backward_simple_rnn_9/Shape:output:02backward_simple_rnn_9/strided_slice/stack:output:04backward_simple_rnn_9/strided_slice/stack_1:output:04backward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_9/zeros/packedPack,backward_simple_rnn_9/strided_slice:output:0-backward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_9/zerosFill+backward_simple_rnn_9/zeros/packed:output:0*backward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_simple_rnn_9/transpose	Transposeinputs-backward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
backward_simple_rnn_9/Shape_1Shape#backward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_9/strided_slice_1StridedSlice&backward_simple_rnn_9/Shape_1:output:04backward_simple_rnn_9/strided_slice_1/stack:output:06backward_simple_rnn_9/strided_slice_1/stack_1:output:06backward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_9/TensorArrayV2TensorListReserve:backward_simple_rnn_9/TensorArrayV2/element_shape:output:0.backward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_9/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¶
backward_simple_rnn_9/ReverseV2	ReverseV2#backward_simple_rnn_9/transpose:y:0-backward_simple_rnn_9/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Kbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   §
=backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_9/ReverseV2:output:0Tbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%backward_simple_rnn_9/strided_slice_2StridedSlice#backward_simple_rnn_9/transpose:y:04backward_simple_rnn_9/strided_slice_2/stack:output:06backward_simple_rnn_9/strided_slice_2/stack_1:output:06backward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_9/simple_rnn_cell_29/MatMulMatMul.backward_simple_rnn_9/strided_slice_2:output:0Fbackward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_9/simple_rnn_cell_29/BiasAddBiasAdd9backward_simple_rnn_9/simple_rnn_cell_29/MatMul:product:0Gbackward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1MatMul$backward_simple_rnn_9/zeros:output:0Hbackward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_9/simple_rnn_cell_29/addAddV29backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd:output:0;backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_9/simple_rnn_cell_29/TanhTanh0backward_simple_rnn_9/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_9/TensorArrayV2_1TensorListReserve<backward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_9/whileWhile1backward_simple_rnn_9/while/loop_counter:output:07backward_simple_rnn_9/while/maximum_iterations:output:0#backward_simple_rnn_9/time:output:0.backward_simple_rnn_9/TensorArrayV2_1:handle:0$backward_simple_rnn_9/zeros:output:0.backward_simple_rnn_9/strided_slice_1:output:0Mbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resourceHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resourceIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
(backward_simple_rnn_9_while_body_4704519*4
cond,R*
(backward_simple_rnn_9_while_cond_4704518*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_9/while:output:3Obackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_9/strided_slice_3StridedSliceAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_9/strided_slice_3/stack:output:06backward_simple_rnn_9/strided_slice_3/stack_1:output:06backward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_9/transpose_1	TransposeAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_9/strided_slice_3:output:0.backward_simple_rnn_9/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?^backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpA^backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp^backward_simple_rnn_9/while?^forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>^forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp@^forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp^forward_simple_rnn_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp2
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp2:
backward_simple_rnn_9/whilebackward_simple_rnn_9/while2
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp2
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp28
forward_simple_rnn_9/whileforward_simple_rnn_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs

¾
'forward_simple_rnn_9_while_cond_4705367F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2H
Dforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705367___redundant_placeholder0_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705367___redundant_placeholder1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705367___redundant_placeholder2_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705367___redundant_placeholder3'
#forward_simple_rnn_9_while_identity
¶
forward_simple_rnn_9/while/LessLess&forward_simple_rnn_9_while_placeholderDforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_9/while/IdentityIdentity#forward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0*(
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
À

Ü
4__inference_simple_rnn_cell_28_layer_call_fn_4707215

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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4702950o
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
ÛA
É
'forward_simple_rnn_9_while_body_4705808F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2E
Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0
}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_9_while_identity)
%forward_simple_rnn_9_while_identity_1)
%forward_simple_rnn_9_while_identity_2)
%forward_simple_rnn_9_while_identity_3)
%forward_simple_rnn_9_while_identity_4C
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp¢Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
Lforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
>forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_9_while_placeholderUforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ò
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_9/while/simple_rnn_cell_28/MatMulMatMulEforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAddBiasAdd>forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul:product:0Lforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1MatMul(forward_simple_rnn_9_while_placeholder_2Mforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_9/while/simple_rnn_cell_28/addAddV2>forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd:output:0@forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_9/while/simple_rnn_cell_28/TanhTanh5forward_simple_rnn_9/while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_9_while_placeholder_1Nforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_9/while/addAddV2&forward_simple_rnn_9_while_placeholder)forward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_9/while/add_1AddV2Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counter+forward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_9/while/IdentityIdentity$forward_simple_rnn_9/while/add_1:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_9/while/Identity_1IdentityHforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_9/while/Identity_2Identity"forward_simple_rnn_9/while/add:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_9/while/Identity_3IdentityOforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_9/while/Identity_4Identity6forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0 ^forward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_9/while/NoOpNoOpE^forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpD^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpF^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0"S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0"W
%forward_simple_rnn_9_while_identity_1.forward_simple_rnn_9/while/Identity_1:output:0"W
%forward_simple_rnn_9_while_identity_2.forward_simple_rnn_9/while/Identity_2:output:0"W
%forward_simple_rnn_9_while_identity_3.forward_simple_rnn_9/while/Identity_3:output:0"W
%forward_simple_rnn_9_while_identity_4.forward_simple_rnn_9/while/Identity_4:output:0" 
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resourceOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resourcePforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resourceNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpDforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpCforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp2
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpEforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
ÝB
è
(backward_simple_rnn_9_while_body_4705696H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2G
Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0
backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_9_while_identity*
&backward_simple_rnn_9_while_identity_1*
&backward_simple_rnn_9_while_identity_2*
&backward_simple_rnn_9_while_identity_3*
&backward_simple_rnn_9_while_identity_4E
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp¢Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
?backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_9_while_placeholderVbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ô
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_9/while/simple_rnn_cell_29/MatMulMatMulFbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAddBiasAdd?backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul:product:0Mbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1MatMul)backward_simple_rnn_9_while_placeholder_2Nbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_9/while/simple_rnn_cell_29/addAddV2?backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd:output:0Abackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_9/while/simple_rnn_cell_29/TanhTanh6backward_simple_rnn_9/while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_9_while_placeholder_1Obackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_9/while/addAddV2'backward_simple_rnn_9_while_placeholder*backward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_9/while/add_1AddV2Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counter,backward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_9/while/IdentityIdentity%backward_simple_rnn_9/while/add_1:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_9/while/Identity_1IdentityJbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_9/while/Identity_2Identity#backward_simple_rnn_9/while/add:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_9/while/Identity_3IdentityPbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_9/while/Identity_4Identity7backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0!^backward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_9/while/NoOpNoOpF^backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpE^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpG^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0"U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0"Y
&backward_simple_rnn_9_while_identity_1/backward_simple_rnn_9/while/Identity_1:output:0"Y
&backward_simple_rnn_9_while_identity_2/backward_simple_rnn_9/while/Identity_2:output:0"Y
&backward_simple_rnn_9_while_identity_3/backward_simple_rnn_9/while/Identity_3:output:0"Y
&backward_simple_rnn_9_while_identity_4/backward_simple_rnn_9/while/Identity_4:output:0"¢
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resourcePbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resourceQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resourceObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0"
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpEbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpDbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp2
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpFbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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

	
9bidirectional_18_backward_simple_rnn_9_while_cond_4704953j
fbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_loop_counterp
lbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_maximum_iterations<
8bidirectional_18_backward_simple_rnn_9_while_placeholder>
:bidirectional_18_backward_simple_rnn_9_while_placeholder_1>
:bidirectional_18_backward_simple_rnn_9_while_placeholder_2l
hbidirectional_18_backward_simple_rnn_9_while_less_bidirectional_18_backward_simple_rnn_9_strided_slice_1
bidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_cond_4704953___redundant_placeholder0
bidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_cond_4704953___redundant_placeholder1
bidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_cond_4704953___redundant_placeholder2
bidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_cond_4704953___redundant_placeholder39
5bidirectional_18_backward_simple_rnn_9_while_identity
þ
1bidirectional_18/backward_simple_rnn_9/while/LessLess8bidirectional_18_backward_simple_rnn_9_while_placeholderhbidirectional_18_backward_simple_rnn_9_while_less_bidirectional_18_backward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: 
5bidirectional_18/backward_simple_rnn_9/while/IdentityIdentity5bidirectional_18/backward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "w
5bidirectional_18_backward_simple_rnn_9_while_identity>bidirectional_18/backward_simple_rnn_9/while/Identity:output:0*(
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

ì
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4707308

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
²
Ñ
(backward_simple_rnn_9_while_cond_4706135H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2J
Fbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4706135___redundant_placeholder0a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4706135___redundant_placeholder1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4706135___redundant_placeholder2a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4706135___redundant_placeholder3(
$backward_simple_rnn_9_while_identity
º
 backward_simple_rnn_9/while/LessLess'backward_simple_rnn_9_while_placeholderFbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_9/while/IdentityIdentity$backward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0*(
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4706865
inputs_0C
1simple_rnn_cell_29_matmul_readvariableop_resource:4@@
2simple_rnn_cell_29_biasadd_readvariableop_resource:@E
3simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_29/BiasAdd/ReadVariableOp¢(simple_rnn_cell_29/MatMul/ReadVariableOp¢*simple_rnn_cell_29/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_29/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_29/BiasAddBiasAdd#simple_rnn_cell_29/MatMul:product:01simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_29/MatMul_1MatMulzeros:output:02simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_29/addAddV2#simple_rnn_cell_29/BiasAdd:output:0%simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_29/TanhTanhsimple_rnn_cell_29/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_29_matmul_readvariableop_resource2simple_rnn_cell_29_biasadd_readvariableop_resource3simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
while_body_4706798*
condR
while_cond_4706797*8
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
NoOpNoOp*^simple_rnn_cell_29/BiasAdd/ReadVariableOp)^simple_rnn_cell_29/MatMul/ReadVariableOp+^simple_rnn_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_29/BiasAdd/ReadVariableOp)simple_rnn_cell_29/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_29/MatMul/ReadVariableOp(simple_rnn_cell_29/MatMul/ReadVariableOp2X
*simple_rnn_cell_29/MatMul_1/ReadVariableOp*simple_rnn_cell_29/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ü-
Ò
while_body_4703817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_29_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_29/MatMul/ReadVariableOp¢0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_29/BiasAddBiasAdd)while/simple_rnn_cell_29/MatMul:product:07while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_29/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_29/addAddV2)while/simple_rnn_cell_29/BiasAdd:output:0+while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_29/TanhTanh while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_29/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_29/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_29/MatMul/ReadVariableOp1^while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_29_matmul_readvariableop_resource9while_simple_rnn_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_29/MatMul/ReadVariableOp.while/simple_rnn_cell_29/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
'forward_simple_rnn_9_while_cond_4704410F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2H
Dforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4704410___redundant_placeholder0_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4704410___redundant_placeholder1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4704410___redundant_placeholder2_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4704410___redundant_placeholder3'
#forward_simple_rnn_9_while_identity
¶
forward_simple_rnn_9/while/LessLess&forward_simple_rnn_9_while_placeholderDforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_9/while/IdentityIdentity#forward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0*(
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
while_cond_4703665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4703665___redundant_placeholder05
1while_while_cond_4703665___redundant_placeholder15
1while_while_cond_4703665___redundant_placeholder25
1while_while_cond_4703665___redundant_placeholder3
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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4702950

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
À

Ü
4__inference_simple_rnn_cell_29_layer_call_fn_4707277

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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4703248o
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
Ù@
Í
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4706977
inputs_0C
1simple_rnn_cell_29_matmul_readvariableop_resource:4@@
2simple_rnn_cell_29_biasadd_readvariableop_resource:@E
3simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_29/BiasAdd/ReadVariableOp¢(simple_rnn_cell_29/MatMul/ReadVariableOp¢*simple_rnn_cell_29/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_29/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_29/BiasAddBiasAdd#simple_rnn_cell_29/MatMul:product:01simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_29/MatMul_1MatMulzeros:output:02simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_29/addAddV2#simple_rnn_cell_29/BiasAdd:output:0%simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_29/TanhTanhsimple_rnn_cell_29/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_29_matmul_readvariableop_resource2simple_rnn_cell_29_biasadd_readvariableop_resource3simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
while_body_4706910*
condR
while_cond_4706909*8
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
NoOpNoOp*^simple_rnn_cell_29/BiasAdd/ReadVariableOp)^simple_rnn_cell_29/MatMul/ReadVariableOp+^simple_rnn_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_29/BiasAdd/ReadVariableOp)simple_rnn_cell_29/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_29/MatMul/ReadVariableOp(simple_rnn_cell_29/MatMul/ReadVariableOp2X
*simple_rnn_cell_29/MatMul_1/ReadVariableOp*simple_rnn_cell_29/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ç
	
8bidirectional_18_forward_simple_rnn_9_while_cond_4705072h
dbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_loop_countern
jbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_maximum_iterations;
7bidirectional_18_forward_simple_rnn_9_while_placeholder=
9bidirectional_18_forward_simple_rnn_9_while_placeholder_1=
9bidirectional_18_forward_simple_rnn_9_while_placeholder_2j
fbidirectional_18_forward_simple_rnn_9_while_less_bidirectional_18_forward_simple_rnn_9_strided_slice_1
}bidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_cond_4705072___redundant_placeholder0
}bidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_cond_4705072___redundant_placeholder1
}bidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_cond_4705072___redundant_placeholder2
}bidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_cond_4705072___redundant_placeholder38
4bidirectional_18_forward_simple_rnn_9_while_identity
ú
0bidirectional_18/forward_simple_rnn_9/while/LessLess7bidirectional_18_forward_simple_rnn_9_while_placeholderfbidirectional_18_forward_simple_rnn_9_while_less_bidirectional_18_forward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: 
4bidirectional_18/forward_simple_rnn_9/while/IdentityIdentity4bidirectional_18/forward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "u
4bidirectional_18_forward_simple_rnn_9_while_identity=bidirectional_18/forward_simple_rnn_9/while/Identity:output:0*(
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
(backward_simple_rnn_9_while_cond_4705695H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2J
Fbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705695___redundant_placeholder0a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705695___redundant_placeholder1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705695___redundant_placeholder2a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705695___redundant_placeholder3(
$backward_simple_rnn_9_while_identity
º
 backward_simple_rnn_9/while/LessLess'backward_simple_rnn_9_while_placeholderFbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_9/while/IdentityIdentity$backward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0*(
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
while_body_4706422
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_28_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_28/MatMul/ReadVariableOp¢0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_28/BiasAddBiasAdd)while/simple_rnn_cell_28/MatMul:product:07while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_28/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_28/addAddV2)while/simple_rnn_cell_28/BiasAdd:output:0+while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_28/TanhTanh while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_28/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_28/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_28/MatMul/ReadVariableOp1^while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_28_matmul_readvariableop_resource9while_simple_rnn_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_28/MatMul/ReadVariableOp.while/simple_rnn_cell_28/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
while_body_4703547
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_28_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_28/MatMul/ReadVariableOp¢0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_28/BiasAddBiasAdd)while/simple_rnn_cell_28/MatMul:product:07while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_28/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_28/addAddV2)while/simple_rnn_cell_28/BiasAdd:output:0+while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_28/TanhTanh while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_28/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_28/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_28/MatMul/ReadVariableOp1^while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_28_matmul_readvariableop_resource9while_simple_rnn_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_28/MatMul/ReadVariableOp.while/simple_rnn_cell_28/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
Ú	
Ã
/__inference_sequential_18_layer_call_fn_4704803

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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704648o
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
§
Í
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705985

inputsX
Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp¢@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp¢backward_simple_rnn_9/while¢>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp¢?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp¢forward_simple_rnn_9/whileP
forward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_9/strided_sliceStridedSlice#forward_simple_rnn_9/Shape:output:01forward_simple_rnn_9/strided_slice/stack:output:03forward_simple_rnn_9/strided_slice/stack_1:output:03forward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_9/zeros/packedPack+forward_simple_rnn_9/strided_slice:output:0,forward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_9/zerosFill*forward_simple_rnn_9/zeros/packed:output:0)forward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_9/transpose	Transposeinputs,forward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4n
forward_simple_rnn_9/Shape_1Shape"forward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_9/strided_slice_1StridedSlice%forward_simple_rnn_9/Shape_1:output:03forward_simple_rnn_9/strided_slice_1/stack:output:05forward_simple_rnn_9/strided_slice_1/stack_1:output:05forward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_9/TensorArrayV2TensorListReserve9forward_simple_rnn_9/TensorArrayV2/element_shape:output:0-forward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
<forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_9/transpose:y:0Sforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$forward_simple_rnn_9/strided_slice_2StridedSlice"forward_simple_rnn_9/transpose:y:03forward_simple_rnn_9/strided_slice_2/stack:output:05forward_simple_rnn_9/strided_slice_2/stack_1:output:05forward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÄ
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_9/simple_rnn_cell_28/MatMulMatMul-forward_simple_rnn_9/strided_slice_2:output:0Eforward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_9/simple_rnn_cell_28/BiasAddBiasAdd8forward_simple_rnn_9/simple_rnn_cell_28/MatMul:product:0Fforward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1MatMul#forward_simple_rnn_9/zeros:output:0Gforward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_9/simple_rnn_cell_28/addAddV28forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd:output:0:forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_9/simple_rnn_cell_28/TanhTanh/forward_simple_rnn_9/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_9/TensorArrayV2_1TensorListReserve;forward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_9/whileWhile0forward_simple_rnn_9/while/loop_counter:output:06forward_simple_rnn_9/while/maximum_iterations:output:0"forward_simple_rnn_9/time:output:0-forward_simple_rnn_9/TensorArrayV2_1:handle:0#forward_simple_rnn_9/zeros:output:0-forward_simple_rnn_9/strided_slice_1:output:0Lforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resourceGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resourceHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
'forward_simple_rnn_9_while_body_4705808*3
cond+R)
'forward_simple_rnn_9_while_cond_4705807*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_9/while:output:3Nforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_9/strided_slice_3StridedSlice@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_9/strided_slice_3/stack:output:05forward_simple_rnn_9/strided_slice_3/stack_1:output:05forward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_9/transpose_1	Transpose@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
backward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_9/strided_sliceStridedSlice$backward_simple_rnn_9/Shape:output:02backward_simple_rnn_9/strided_slice/stack:output:04backward_simple_rnn_9/strided_slice/stack_1:output:04backward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_9/zeros/packedPack,backward_simple_rnn_9/strided_slice:output:0-backward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_9/zerosFill+backward_simple_rnn_9/zeros/packed:output:0*backward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_simple_rnn_9/transpose	Transposeinputs-backward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
backward_simple_rnn_9/Shape_1Shape#backward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_9/strided_slice_1StridedSlice&backward_simple_rnn_9/Shape_1:output:04backward_simple_rnn_9/strided_slice_1/stack:output:06backward_simple_rnn_9/strided_slice_1/stack_1:output:06backward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_9/TensorArrayV2TensorListReserve:backward_simple_rnn_9/TensorArrayV2/element_shape:output:0.backward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_9/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¶
backward_simple_rnn_9/ReverseV2	ReverseV2#backward_simple_rnn_9/transpose:y:0-backward_simple_rnn_9/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Kbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   §
=backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_9/ReverseV2:output:0Tbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%backward_simple_rnn_9/strided_slice_2StridedSlice#backward_simple_rnn_9/transpose:y:04backward_simple_rnn_9/strided_slice_2/stack:output:06backward_simple_rnn_9/strided_slice_2/stack_1:output:06backward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_9/simple_rnn_cell_29/MatMulMatMul.backward_simple_rnn_9/strided_slice_2:output:0Fbackward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_9/simple_rnn_cell_29/BiasAddBiasAdd9backward_simple_rnn_9/simple_rnn_cell_29/MatMul:product:0Gbackward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1MatMul$backward_simple_rnn_9/zeros:output:0Hbackward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_9/simple_rnn_cell_29/addAddV29backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd:output:0;backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_9/simple_rnn_cell_29/TanhTanh0backward_simple_rnn_9/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_9/TensorArrayV2_1TensorListReserve<backward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_9/whileWhile1backward_simple_rnn_9/while/loop_counter:output:07backward_simple_rnn_9/while/maximum_iterations:output:0#backward_simple_rnn_9/time:output:0.backward_simple_rnn_9/TensorArrayV2_1:handle:0$backward_simple_rnn_9/zeros:output:0.backward_simple_rnn_9/strided_slice_1:output:0Mbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resourceHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resourceIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
(backward_simple_rnn_9_while_body_4705916*4
cond,R*
(backward_simple_rnn_9_while_cond_4705915*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_9/while:output:3Obackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_9/strided_slice_3StridedSliceAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_9/strided_slice_3/stack:output:06backward_simple_rnn_9/strided_slice_3/stack_1:output:06backward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_9/transpose_1	TransposeAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_9/strided_slice_3:output:0.backward_simple_rnn_9/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?^backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpA^backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp^backward_simple_rnn_9/while?^forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>^forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp@^forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp^forward_simple_rnn_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp2
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp2:
backward_simple_rnn_9/whilebackward_simple_rnn_9/while2
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp2
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp28
forward_simple_rnn_9/whileforward_simple_rnn_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ð
Ó
"__inference__wrapped_model_4702902
bidirectional_18_inputw
esequential_18_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource:4@t
fsequential_18_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource:@y
gsequential_18_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@x
fsequential_18_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource:4@u
gsequential_18_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource:@z
hsequential_18_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@H
5sequential_18_dense_18_matmul_readvariableop_resource:	D
6sequential_18_dense_18_biasadd_readvariableop_resource:
identity¢^sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢]sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp¢_sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp¢:sequential_18/bidirectional_18/backward_simple_rnn_9/while¢]sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢\sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp¢^sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp¢9sequential_18/bidirectional_18/forward_simple_rnn_9/while¢-sequential_18/dense_18/BiasAdd/ReadVariableOp¢,sequential_18/dense_18/MatMul/ReadVariableOp
9sequential_18/bidirectional_18/forward_simple_rnn_9/ShapeShapebidirectional_18_input*
T0*
_output_shapes
:
Gsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Isequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Isequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Õ
Asequential_18/bidirectional_18/forward_simple_rnn_9/strided_sliceStridedSliceBsequential_18/bidirectional_18/forward_simple_rnn_9/Shape:output:0Psequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice/stack:output:0Rsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice/stack_1:output:0Rsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bsequential_18/bidirectional_18/forward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
@sequential_18/bidirectional_18/forward_simple_rnn_9/zeros/packedPackJsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice:output:0Ksequential_18/bidirectional_18/forward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
?sequential_18/bidirectional_18/forward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
9sequential_18/bidirectional_18/forward_simple_rnn_9/zerosFillIsequential_18/bidirectional_18/forward_simple_rnn_9/zeros/packed:output:0Hsequential_18/bidirectional_18/forward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Bsequential_18/bidirectional_18/forward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          å
=sequential_18/bidirectional_18/forward_simple_rnn_9/transpose	Transposebidirectional_18_inputKsequential_18/bidirectional_18/forward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4¬
;sequential_18/bidirectional_18/forward_simple_rnn_9/Shape_1ShapeAsequential_18/bidirectional_18/forward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:
Isequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ß
Csequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_1StridedSliceDsequential_18/bidirectional_18/forward_simple_rnn_9/Shape_1:output:0Rsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_1/stack:output:0Tsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_1:output:0Tsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Osequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÐ
Asequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2TensorListReserveXsequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2/element_shape:output:0Lsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒº
isequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ü
[sequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorAsequential_18/bidirectional_18/forward_simple_rnn_9/transpose:y:0rsequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:í
Csequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_2StridedSliceAsequential_18/bidirectional_18/forward_simple_rnn_9/transpose:y:0Rsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_2/stack:output:0Tsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_1:output:0Tsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
\sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpesequential_18_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0½
Msequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMulMatMulLsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_2:output:0dsequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
]sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpfsequential_18_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ë
Nsequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAddBiasAddWsequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul:product:0esequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
^sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpgsequential_18_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0·
Osequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1MatMulBsequential_18/bidirectional_18/forward_simple_rnn_9/zeros:output:0fsequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¹
Jsequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/addAddV2Wsequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd:output:0Ysequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Õ
Ksequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/TanhTanhNsequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
Qsequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Psequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :á
Csequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1TensorListReserveZsequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0Ysequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒz
8sequential_18/bidirectional_18/forward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Lsequential_18/bidirectional_18/forward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Fsequential_18/bidirectional_18/forward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
9sequential_18/bidirectional_18/forward_simple_rnn_9/whileWhileOsequential_18/bidirectional_18/forward_simple_rnn_9/while/loop_counter:output:0Usequential_18/bidirectional_18/forward_simple_rnn_9/while/maximum_iterations:output:0Asequential_18/bidirectional_18/forward_simple_rnn_9/time:output:0Lsequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1:handle:0Bsequential_18/bidirectional_18/forward_simple_rnn_9/zeros:output:0Lsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_1:output:0ksequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0esequential_18_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resourcefsequential_18_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resourcegsequential_18_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
Fsequential_18_bidirectional_18_forward_simple_rnn_9_while_body_4702718*R
condJRH
Fsequential_18_bidirectional_18_forward_simple_rnn_9_while_cond_4702717*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations µ
dsequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ò
Vsequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStackBsequential_18/bidirectional_18/forward_simple_rnn_9/while:output:3msequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
Isequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Ksequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Csequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_3StridedSlice_sequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0Rsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_3/stack:output:0Tsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_1:output:0Tsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
Dsequential_18/bidirectional_18/forward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ²
?sequential_18/bidirectional_18/forward_simple_rnn_9/transpose_1	Transpose_sequential_18/bidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0Msequential_18/bidirectional_18/forward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
:sequential_18/bidirectional_18/backward_simple_rnn_9/ShapeShapebidirectional_18_input*
T0*
_output_shapes
:
Hsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ú
Bsequential_18/bidirectional_18/backward_simple_rnn_9/strided_sliceStridedSliceCsequential_18/bidirectional_18/backward_simple_rnn_9/Shape:output:0Qsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice/stack:output:0Ssequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice/stack_1:output:0Ssequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Csequential_18/bidirectional_18/backward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
Asequential_18/bidirectional_18/backward_simple_rnn_9/zeros/packedPackKsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice:output:0Lsequential_18/bidirectional_18/backward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
@sequential_18/bidirectional_18/backward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
:sequential_18/bidirectional_18/backward_simple_rnn_9/zerosFillJsequential_18/bidirectional_18/backward_simple_rnn_9/zeros/packed:output:0Isequential_18/bidirectional_18/backward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Csequential_18/bidirectional_18/backward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ç
>sequential_18/bidirectional_18/backward_simple_rnn_9/transpose	Transposebidirectional_18_inputLsequential_18/bidirectional_18/backward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4®
<sequential_18/bidirectional_18/backward_simple_rnn_9/Shape_1ShapeBsequential_18/bidirectional_18/backward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:
Jsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ä
Dsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_1StridedSliceEsequential_18/bidirectional_18/backward_simple_rnn_9/Shape_1:output:0Ssequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_1/stack:output:0Usequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_1:output:0Usequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Psequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
Bsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2TensorListReserveYsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2/element_shape:output:0Msequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Csequential_18/bidirectional_18/backward_simple_rnn_9/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
>sequential_18/bidirectional_18/backward_simple_rnn_9/ReverseV2	ReverseV2Bsequential_18/bidirectional_18/backward_simple_rnn_9/transpose:y:0Lsequential_18/bidirectional_18/backward_simple_rnn_9/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4»
jsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
\sequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGsequential_18/bidirectional_18/backward_simple_rnn_9/ReverseV2:output:0ssequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Dsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_2StridedSliceBsequential_18/bidirectional_18/backward_simple_rnn_9/transpose:y:0Ssequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_2/stack:output:0Usequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_1:output:0Usequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
]sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpfsequential_18_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0À
Nsequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMulMatMulMsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_2:output:0esequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
^sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpgsequential_18_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Î
Osequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAddBiasAddXsequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul:product:0fsequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
_sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOphsequential_18_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0º
Psequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1MatMulCsequential_18/bidirectional_18/backward_simple_rnn_9/zeros:output:0gsequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
Ksequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/addAddV2Xsequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd:output:0Zsequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@×
Lsequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/TanhTanhOsequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
Rsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Qsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ä
Dsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1TensorListReserve[sequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0Zsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ{
9sequential_18/bidirectional_18/backward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Msequential_18/bidirectional_18/backward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Gsequential_18/bidirectional_18/backward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
:sequential_18/bidirectional_18/backward_simple_rnn_9/whileWhilePsequential_18/bidirectional_18/backward_simple_rnn_9/while/loop_counter:output:0Vsequential_18/bidirectional_18/backward_simple_rnn_9/while/maximum_iterations:output:0Bsequential_18/bidirectional_18/backward_simple_rnn_9/time:output:0Msequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1:handle:0Csequential_18/bidirectional_18/backward_simple_rnn_9/zeros:output:0Msequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_1:output:0lsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0fsequential_18_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resourcegsequential_18_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resourcehsequential_18_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
Gsequential_18_bidirectional_18_backward_simple_rnn_9_while_body_4702826*S
condKRI
Gsequential_18_bidirectional_18_backward_simple_rnn_9_while_cond_4702825*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¶
esequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   õ
Wsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStackCsequential_18/bidirectional_18/backward_simple_rnn_9/while:output:3nsequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
Jsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Lsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_3StridedSlice`sequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0Ssequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_3/stack:output:0Usequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_1:output:0Usequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
Esequential_18/bidirectional_18/backward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
@sequential_18/bidirectional_18/backward_simple_rnn_9/transpose_1	Transpose`sequential_18/bidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0Nsequential_18/bidirectional_18/backward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
*sequential_18/bidirectional_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¿
%sequential_18/bidirectional_18/concatConcatV2Lsequential_18/bidirectional_18/forward_simple_rnn_9/strided_slice_3:output:0Msequential_18/bidirectional_18/backward_simple_rnn_9/strided_slice_3:output:03sequential_18/bidirectional_18/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,sequential_18/dense_18/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¿
sequential_18/dense_18/MatMulMatMul.sequential_18/bidirectional_18/concat:output:04sequential_18/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_18/dense_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_18/dense_18/BiasAddBiasAdd'sequential_18/dense_18/MatMul:product:05sequential_18/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_18/dense_18/SoftmaxSoftmax'sequential_18/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_18/dense_18/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
NoOpNoOp_^sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp^^sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp`^sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp;^sequential_18/bidirectional_18/backward_simple_rnn_9/while^^sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp]^sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp_^sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:^sequential_18/bidirectional_18/forward_simple_rnn_9/while.^sequential_18/dense_18/BiasAdd/ReadVariableOp-^sequential_18/dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2À
^sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp^sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp2¾
]sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp]sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp2Â
_sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp_sequential_18/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp2x
:sequential_18/bidirectional_18/backward_simple_rnn_9/while:sequential_18/bidirectional_18/backward_simple_rnn_9/while2¾
]sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp]sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp2¼
\sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp\sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp2À
^sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp^sequential_18/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp2v
9sequential_18/bidirectional_18/forward_simple_rnn_9/while9sequential_18/bidirectional_18/forward_simple_rnn_9/while2^
-sequential_18/dense_18/BiasAdd/ReadVariableOp-sequential_18/dense_18/BiasAdd/ReadVariableOp2\
,sequential_18/dense_18/MatMul/ReadVariableOp,sequential_18/dense_18/MatMul/ReadVariableOp:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_18_input
½"
ß
while_body_4703125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_28_4703147_0:4@0
"while_simple_rnn_cell_28_4703149_0:@4
"while_simple_rnn_cell_28_4703151_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_28_4703147:4@.
 while_simple_rnn_cell_28_4703149:@2
 while_simple_rnn_cell_28_4703151:@@¢0while/simple_rnn_cell_28/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_28/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_28_4703147_0"while_simple_rnn_cell_28_4703149_0"while_simple_rnn_cell_28_4703151_0*
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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4703072r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_28/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_28/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_28_4703147"while_simple_rnn_cell_28_4703147_0"F
 while_simple_rnn_cell_28_4703149"while_simple_rnn_cell_28_4703149_0"F
 while_simple_rnn_cell_28_4703151"while_simple_rnn_cell_28_4703151_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_28/StatefulPartitionedCall0while/simple_rnn_cell_28/StatefulPartitionedCall: 
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
'forward_simple_rnn_9_while_cond_4704110F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2H
Dforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4704110___redundant_placeholder0_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4704110___redundant_placeholder1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4704110___redundant_placeholder2_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4704110___redundant_placeholder3'
#forward_simple_rnn_9_while_identity
¶
forward_simple_rnn_9/while/LessLess&forward_simple_rnn_9_while_placeholderDforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_9/while/IdentityIdentity#forward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0*(
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
7__inference_backward_simple_rnn_9_layer_call_fn_4706731
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703489o
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
ÔB
è
(backward_simple_rnn_9_while_body_4706136H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2G
Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0
backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_9_while_identity*
&backward_simple_rnn_9_while_identity_1*
&backward_simple_rnn_9_while_identity_2*
&backward_simple_rnn_9_while_identity_3*
&backward_simple_rnn_9_while_identity_4E
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp¢Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_9_while_placeholderVbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_9/while/simple_rnn_cell_29/MatMulMatMulFbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAddBiasAdd?backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul:product:0Mbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1MatMul)backward_simple_rnn_9_while_placeholder_2Nbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_9/while/simple_rnn_cell_29/addAddV2?backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd:output:0Abackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_9/while/simple_rnn_cell_29/TanhTanh6backward_simple_rnn_9/while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_9_while_placeholder_1Obackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_9/while/addAddV2'backward_simple_rnn_9_while_placeholder*backward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_9/while/add_1AddV2Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counter,backward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_9/while/IdentityIdentity%backward_simple_rnn_9/while/add_1:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_9/while/Identity_1IdentityJbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_9/while/Identity_2Identity#backward_simple_rnn_9/while/add:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_9/while/Identity_3IdentityPbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_9/while/Identity_4Identity7backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0!^backward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_9/while/NoOpNoOpF^backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpE^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpG^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0"U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0"Y
&backward_simple_rnn_9_while_identity_1/backward_simple_rnn_9/while/Identity_1:output:0"Y
&backward_simple_rnn_9_while_identity_2/backward_simple_rnn_9/while/Identity_2:output:0"Y
&backward_simple_rnn_9_while_identity_3/backward_simple_rnn_9/while/Identity_3:output:0"Y
&backward_simple_rnn_9_while_identity_4/backward_simple_rnn_9/while/Identity_4:output:0"¢
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resourcePbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resourceQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resourceObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0"
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpEbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpDbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp2
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpFbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
2__inference_bidirectional_18_layer_call_fn_4705325

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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704588p
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
æ
æ
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704047

inputs.
forward_simple_rnn_9_4704030:4@*
forward_simple_rnn_9_4704032:@.
forward_simple_rnn_9_4704034:@@/
backward_simple_rnn_9_4704037:4@+
backward_simple_rnn_9_4704039:@/
backward_simple_rnn_9_4704041:@@
identity¢-backward_simple_rnn_9/StatefulPartitionedCall¢,forward_simple_rnn_9/StatefulPartitionedCallÆ
,forward_simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_9_4704030forward_simple_rnn_9_4704032forward_simple_rnn_9_4704034*
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4704016Ë
-backward_simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_9_4704037backward_simple_rnn_9_4704039backward_simple_rnn_9_4704041*
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703884M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ó
concatConcatV25forward_simple_rnn_9/StatefulPartitionedCall:output:06backward_simple_rnn_9/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp.^backward_simple_rnn_9/StatefulPartitionedCall-^forward_simple_rnn_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_simple_rnn_9/StatefulPartitionedCall-backward_simple_rnn_9/StatefulPartitionedCall2\
,forward_simple_rnn_9/StatefulPartitionedCall,forward_simple_rnn_9/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
ê
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4703072

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
éR
ì
 __inference__traced_save_4707447
file_prefix.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop^
Zsavev2_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_read_readvariableoph
dsavev2_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_read_readvariableop\
Xsavev2_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_read_readvariableop_
[savev2_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_read_readvariableopi
esavev2_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_read_readvariableop]
Ysavev2_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableope
asavev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_m_read_readvariableopo
ksavev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_m_read_readvariableopc
_savev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_m_read_readvariableopf
bsavev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_m_read_readvariableopp
lsavev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_m_read_readvariableopd
`savev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableope
asavev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_v_read_readvariableopo
ksavev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_v_read_readvariableopc
_savev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_v_read_readvariableopf
bsavev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_v_read_readvariableopp
lsavev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_v_read_readvariableopd
`savev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableopZsavev2_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_read_readvariableopdsavev2_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_read_readvariableopXsavev2_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_read_readvariableop[savev2_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_read_readvariableopesavev2_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_read_readvariableopYsavev2_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableopasavev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_m_read_readvariableopksavev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_m_read_readvariableop_savev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_m_read_readvariableopbsavev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_m_read_readvariableoplsavev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_m_read_readvariableop`savev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableopasavev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_v_read_readvariableopksavev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_v_read_readvariableop_savev2_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_v_read_readvariableopbsavev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_v_read_readvariableoplsavev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_v_read_readvariableop`savev2_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
²
Ñ
(backward_simple_rnn_9_while_cond_4705915H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2J
Fbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705915___redundant_placeholder0a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705915___redundant_placeholder1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705915___redundant_placeholder2a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705915___redundant_placeholder3(
$backward_simple_rnn_9_while_identity
º
 backward_simple_rnn_9/while/LessLess'backward_simple_rnn_9_while_placeholderFbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_9/while/IdentityIdentity$backward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0*(
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
ü-
Ò
while_body_4706642
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_28_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_28/MatMul/ReadVariableOp¢0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_28/BiasAddBiasAdd)while/simple_rnn_cell_28/MatMul:product:07while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_28/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_28/addAddV2)while/simple_rnn_cell_28/BiasAdd:output:0+while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_28/TanhTanh while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_28/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_28/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_28/MatMul/ReadVariableOp1^while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_28_matmul_readvariableop_resource9while_simple_rnn_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_28/MatMul/ReadVariableOp.while/simple_rnn_cell_28/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
while_cond_4703546
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4703546___redundant_placeholder05
1while_while_cond_4703546___redundant_placeholder15
1while_while_cond_4703546___redundant_placeholder25
1while_while_cond_4703546___redundant_placeholder3
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
ú>
Ì
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706379
inputs_0C
1simple_rnn_cell_28_matmul_readvariableop_resource:4@@
2simple_rnn_cell_28_biasadd_readvariableop_resource:@E
3simple_rnn_cell_28_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_28/BiasAdd/ReadVariableOp¢(simple_rnn_cell_28/MatMul/ReadVariableOp¢*simple_rnn_cell_28/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_28/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_28/BiasAddBiasAdd#simple_rnn_cell_28/MatMul:product:01simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_28/MatMul_1MatMulzeros:output:02simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_28/addAddV2#simple_rnn_cell_28/BiasAdd:output:0%simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_28/TanhTanhsimple_rnn_cell_28/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_28_matmul_readvariableop_resource2simple_rnn_cell_28_biasadd_readvariableop_resource3simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
while_body_4706312*
condR
while_cond_4706311*8
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
NoOpNoOp*^simple_rnn_cell_28/BiasAdd/ReadVariableOp)^simple_rnn_cell_28/MatMul/ReadVariableOp+^simple_rnn_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_28/BiasAdd/ReadVariableOp)simple_rnn_cell_28/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_28/MatMul/ReadVariableOp(simple_rnn_cell_28/MatMul/ReadVariableOp2X
*simple_rnn_cell_28/MatMul_1/ReadVariableOp*simple_rnn_cell_28/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ß
¯
while_cond_4707021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4707021___redundant_placeholder05
1while_while_cond_4707021___redundant_placeholder15
1while_while_cond_4707021___redundant_placeholder25
1while_while_cond_4707021___redundant_placeholder3
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
'forward_simple_rnn_9_while_cond_4705807F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2H
Dforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705807___redundant_placeholder0_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705807___redundant_placeholder1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705807___redundant_placeholder2_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4705807___redundant_placeholder3'
#forward_simple_rnn_9_while_identity
¶
forward_simple_rnn_9/while/LessLess&forward_simple_rnn_9_while_placeholderDforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_9/while/IdentityIdentity#forward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0*(
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706236
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4703028o
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


Ó
/__inference_sequential_18_layer_call_fn_4704339
bidirectional_18_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704320o
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
_user_specified_namebidirectional_18_input
§Ñ
ï
J__inference_sequential_18_layer_call_and_return_conditional_losses_4705257

inputsi
Wbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource:4@f
Xbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource:@k
Ybidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@j
Xbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource:4@g
Ybidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource:@l
Zbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@:
'dense_18_matmul_readvariableop_resource:	6
(dense_18_biasadd_readvariableop_resource:
identity¢Pbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Obidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp¢Qbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp¢,bidirectional_18/backward_simple_rnn_9/while¢Obidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Nbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp¢Pbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp¢+bidirectional_18/forward_simple_rnn_9/while¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOpa
+bidirectional_18/forward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:
9bidirectional_18/forward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;bidirectional_18/forward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;bidirectional_18/forward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3bidirectional_18/forward_simple_rnn_9/strided_sliceStridedSlice4bidirectional_18/forward_simple_rnn_9/Shape:output:0Bbidirectional_18/forward_simple_rnn_9/strided_slice/stack:output:0Dbidirectional_18/forward_simple_rnn_9/strided_slice/stack_1:output:0Dbidirectional_18/forward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4bidirectional_18/forward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@å
2bidirectional_18/forward_simple_rnn_9/zeros/packedPack<bidirectional_18/forward_simple_rnn_9/strided_slice:output:0=bidirectional_18/forward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:v
1bidirectional_18/forward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Þ
+bidirectional_18/forward_simple_rnn_9/zerosFill;bidirectional_18/forward_simple_rnn_9/zeros/packed:output:0:bidirectional_18/forward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4bidirectional_18/forward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¹
/bidirectional_18/forward_simple_rnn_9/transpose	Transposeinputs=bidirectional_18/forward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
-bidirectional_18/forward_simple_rnn_9/Shape_1Shape3bidirectional_18/forward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:
;bidirectional_18/forward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_18/forward_simple_rnn_9/strided_slice_1StridedSlice6bidirectional_18/forward_simple_rnn_9/Shape_1:output:0Dbidirectional_18/forward_simple_rnn_9/strided_slice_1/stack:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_1:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Abidirectional_18/forward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
3bidirectional_18/forward_simple_rnn_9/TensorArrayV2TensorListReserveJbidirectional_18/forward_simple_rnn_9/TensorArrayV2/element_shape:output:0>bidirectional_18/forward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¬
[bidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ò
Mbidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor3bidirectional_18/forward_simple_rnn_9/transpose:y:0dbidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;bidirectional_18/forward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:§
5bidirectional_18/forward_simple_rnn_9/strided_slice_2StridedSlice3bidirectional_18/forward_simple_rnn_9/transpose:y:0Dbidirectional_18/forward_simple_rnn_9/strided_slice_2/stack:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_1:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskæ
Nbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpWbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
?bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMulMatMul>bidirectional_18/forward_simple_rnn_9/strided_slice_2:output:0Vbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Obidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpXbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¡
@bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAddBiasAddIbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul:product:0Wbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
Pbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpYbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Abidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1MatMul4bidirectional_18/forward_simple_rnn_9/zeros:output:0Xbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
<bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/addAddV2Ibidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd:output:0Kbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¹
=bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/TanhTanh@bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Cbidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Bbidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :·
5bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1TensorListReserveLbidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0Kbidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
*bidirectional_18/forward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 
>bidirectional_18/forward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿz
8bidirectional_18/forward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ë	
+bidirectional_18/forward_simple_rnn_9/whileWhileAbidirectional_18/forward_simple_rnn_9/while/loop_counter:output:0Gbidirectional_18/forward_simple_rnn_9/while/maximum_iterations:output:03bidirectional_18/forward_simple_rnn_9/time:output:0>bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1:handle:04bidirectional_18/forward_simple_rnn_9/zeros:output:0>bidirectional_18/forward_simple_rnn_9/strided_slice_1:output:0]bidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Wbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resourceXbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resourceYbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
8bidirectional_18_forward_simple_rnn_9_while_body_4705073*D
cond<R:
8bidirectional_18_forward_simple_rnn_9_while_cond_4705072*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations §
Vbidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   È
Hbidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack4bidirectional_18/forward_simple_rnn_9/while:output:3_bidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
;bidirectional_18/forward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
=bidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Å
5bidirectional_18/forward_simple_rnn_9/strided_slice_3StridedSliceQbidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0Dbidirectional_18/forward_simple_rnn_9/strided_slice_3/stack:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_1:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
6bidirectional_18/forward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
1bidirectional_18/forward_simple_rnn_9/transpose_1	TransposeQbidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0?bidirectional_18/forward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
,bidirectional_18/backward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:
:bidirectional_18/backward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<bidirectional_18/backward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<bidirectional_18/backward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4bidirectional_18/backward_simple_rnn_9/strided_sliceStridedSlice5bidirectional_18/backward_simple_rnn_9/Shape:output:0Cbidirectional_18/backward_simple_rnn_9/strided_slice/stack:output:0Ebidirectional_18/backward_simple_rnn_9/strided_slice/stack_1:output:0Ebidirectional_18/backward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5bidirectional_18/backward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@è
3bidirectional_18/backward_simple_rnn_9/zeros/packedPack=bidirectional_18/backward_simple_rnn_9/strided_slice:output:0>bidirectional_18/backward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:w
2bidirectional_18/backward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    á
,bidirectional_18/backward_simple_rnn_9/zerosFill<bidirectional_18/backward_simple_rnn_9/zeros/packed:output:0;bidirectional_18/backward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5bidirectional_18/backward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
0bidirectional_18/backward_simple_rnn_9/transpose	Transposeinputs>bidirectional_18/backward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
.bidirectional_18/backward_simple_rnn_9/Shape_1Shape4bidirectional_18/backward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:
<bidirectional_18/backward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_18/backward_simple_rnn_9/strided_slice_1StridedSlice7bidirectional_18/backward_simple_rnn_9/Shape_1:output:0Ebidirectional_18/backward_simple_rnn_9/strided_slice_1/stack:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_1:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bbidirectional_18/backward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
4bidirectional_18/backward_simple_rnn_9/TensorArrayV2TensorListReserveKbidirectional_18/backward_simple_rnn_9/TensorArrayV2/element_shape:output:0?bidirectional_18/backward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5bidirectional_18/backward_simple_rnn_9/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: é
0bidirectional_18/backward_simple_rnn_9/ReverseV2	ReverseV24bidirectional_18/backward_simple_rnn_9/transpose:y:0>bidirectional_18/backward_simple_rnn_9/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4­
\bidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ú
Nbidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor9bidirectional_18/backward_simple_rnn_9/ReverseV2:output:0ebidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<bidirectional_18/backward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
6bidirectional_18/backward_simple_rnn_9/strided_slice_2StridedSlice4bidirectional_18/backward_simple_rnn_9/transpose:y:0Ebidirectional_18/backward_simple_rnn_9/strided_slice_2/stack:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_1:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskè
Obidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpXbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
@bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMulMatMul?bidirectional_18/backward_simple_rnn_9/strided_slice_2:output:0Wbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Pbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpYbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Abidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAddBiasAddJbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul:product:0Xbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ì
Qbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpZbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Bbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1MatMul5bidirectional_18/backward_simple_rnn_9/zeros:output:0Ybidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
=bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/addAddV2Jbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd:output:0Lbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
>bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/TanhTanhAbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dbidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Cbidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :º
6bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1TensorListReserveMbidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0Lbidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
+bidirectional_18/backward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 
?bidirectional_18/backward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ{
9bidirectional_18/backward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø	
,bidirectional_18/backward_simple_rnn_9/whileWhileBbidirectional_18/backward_simple_rnn_9/while/loop_counter:output:0Hbidirectional_18/backward_simple_rnn_9/while/maximum_iterations:output:04bidirectional_18/backward_simple_rnn_9/time:output:0?bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1:handle:05bidirectional_18/backward_simple_rnn_9/zeros:output:0?bidirectional_18/backward_simple_rnn_9/strided_slice_1:output:0^bidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Xbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resourceYbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resourceZbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
9bidirectional_18_backward_simple_rnn_9_while_body_4705181*E
cond=R;
9bidirectional_18_backward_simple_rnn_9_while_cond_4705180*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¨
Wbidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
Ibidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack5bidirectional_18/backward_simple_rnn_9/while:output:3`bidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
<bidirectional_18/backward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
>bidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ê
6bidirectional_18/backward_simple_rnn_9/strided_slice_3StridedSliceRbidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0Ebidirectional_18/backward_simple_rnn_9/strided_slice_3/stack:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_1:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
7bidirectional_18/backward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
2bidirectional_18/backward_simple_rnn_9/transpose_1	TransposeRbidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional_18/backward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
bidirectional_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
bidirectional_18/concatConcatV2>bidirectional_18/forward_simple_rnn_9/strided_slice_3:output:0?bidirectional_18/backward_simple_rnn_9/strided_slice_3:output:0%bidirectional_18/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_18/MatMulMatMul bidirectional_18/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_18/SoftmaxSoftmaxdense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_18/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
NoOpNoOpQ^bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpP^bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpR^bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp-^bidirectional_18/backward_simple_rnn_9/whileP^bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpO^bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpQ^bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp,^bidirectional_18/forward_simple_rnn_9/while ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2¤
Pbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpPbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp2¢
Obidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpObidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp2¦
Qbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpQbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp2\
,bidirectional_18/backward_simple_rnn_9/while,bidirectional_18/backward_simple_rnn_9/while2¢
Obidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpObidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp2 
Nbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpNbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp2¤
Pbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpPbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp2Z
+bidirectional_18/forward_simple_rnn_9/while+bidirectional_18/forward_simple_rnn_9/while2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
?
Ê
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706599

inputsC
1simple_rnn_cell_28_matmul_readvariableop_resource:4@@
2simple_rnn_cell_28_biasadd_readvariableop_resource:@E
3simple_rnn_cell_28_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_28/BiasAdd/ReadVariableOp¢(simple_rnn_cell_28/MatMul/ReadVariableOp¢*simple_rnn_cell_28/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_28/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_28/BiasAddBiasAdd#simple_rnn_cell_28/MatMul:product:01simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_28/MatMul_1MatMulzeros:output:02simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_28/addAddV2#simple_rnn_cell_28/BiasAdd:output:0%simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_28/TanhTanhsimple_rnn_cell_28/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_28_matmul_readvariableop_resource2simple_rnn_cell_28_biasadd_readvariableop_resource3simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
while_body_4706532*
condR
while_cond_4706531*8
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
NoOpNoOp*^simple_rnn_cell_28/BiasAdd/ReadVariableOp)^simple_rnn_cell_28/MatMul/ReadVariableOp+^simple_rnn_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_28/BiasAdd/ReadVariableOp)simple_rnn_cell_28/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_28/MatMul/ReadVariableOp(simple_rnn_cell_28/MatMul/ReadVariableOp2X
*simple_rnn_cell_28/MatMul_1/ReadVariableOp*simple_rnn_cell_28/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
'forward_simple_rnn_9_while_cond_4706027F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2H
Dforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4706027___redundant_placeholder0_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4706027___redundant_placeholder1_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4706027___redundant_placeholder2_
[forward_simple_rnn_9_while_forward_simple_rnn_9_while_cond_4706027___redundant_placeholder3'
#forward_simple_rnn_9_while_identity
¶
forward_simple_rnn_9/while/LessLess&forward_simple_rnn_9_while_placeholderDforward_simple_rnn_9_while_less_forward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_9/while/IdentityIdentity#forward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0*(
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
	

2__inference_bidirectional_18_layer_call_fn_4705308

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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704288p
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
ù^
õ
Fsequential_18_bidirectional_18_forward_simple_rnn_9_while_body_4702718
sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_loop_counter
sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_maximum_iterationsI
Esequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholderK
Gsequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholder_1K
Gsequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholder_2
sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_strided_slice_1_0À
»sequential_18_bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0
msequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@|
nsequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@
osequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@F
Bsequential_18_bidirectional_18_forward_simple_rnn_9_while_identityH
Dsequential_18_bidirectional_18_forward_simple_rnn_9_while_identity_1H
Dsequential_18_bidirectional_18_forward_simple_rnn_9_while_identity_2H
Dsequential_18_bidirectional_18_forward_simple_rnn_9_while_identity_3H
Dsequential_18_bidirectional_18_forward_simple_rnn_9_while_identity_4
}sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_strided_slice_1¾
¹sequential_18_bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor}
ksequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource:4@z
lsequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource:@
msequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢csequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢bsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp¢dsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp¼
ksequential_18/bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   «
]sequential_18/bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem»sequential_18_bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0Esequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholdertsequential_18/bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0
bsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpmsequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0á
Ssequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMulMatMuldsequential_18/bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0jsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
csequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpnsequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ý
Tsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAddBiasAdd]sequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul:product:0ksequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOposequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0È
Usequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1MatMulGsequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholder_2lsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ë
Psequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/addAddV2]sequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd:output:0_sequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@á
Qsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/TanhTanhTsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
dsequential_18/bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Â
^sequential_18/bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemGsequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholder_1msequential_18/bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0Usequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ
?sequential_18/bidirectional_18/forward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ø
=sequential_18/bidirectional_18/forward_simple_rnn_9/while/addAddV2Esequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholderHsequential_18/bidirectional_18/forward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: 
Asequential_18/bidirectional_18/forward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¸
?sequential_18/bidirectional_18/forward_simple_rnn_9/while/add_1AddV2sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_loop_counterJsequential_18/bidirectional_18/forward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: õ
Bsequential_18/bidirectional_18/forward_simple_rnn_9/while/IdentityIdentityCsequential_18/bidirectional_18/forward_simple_rnn_9/while/add_1:z:0?^sequential_18/bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: »
Dsequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity_1Identitysequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_maximum_iterations?^sequential_18/bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: õ
Dsequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity_2IdentityAsequential_18/bidirectional_18/forward_simple_rnn_9/while/add:z:0?^sequential_18/bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ¢
Dsequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity_3Identitynsequential_18/bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0?^sequential_18/bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
Dsequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity_4IdentityUsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0?^sequential_18/bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²
>sequential_18/bidirectional_18/forward_simple_rnn_9/while/NoOpNoOpd^sequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpc^sequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpe^sequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Bsequential_18_bidirectional_18_forward_simple_rnn_9_while_identityKsequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity:output:0"
Dsequential_18_bidirectional_18_forward_simple_rnn_9_while_identity_1Msequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity_1:output:0"
Dsequential_18_bidirectional_18_forward_simple_rnn_9_while_identity_2Msequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity_2:output:0"
Dsequential_18_bidirectional_18_forward_simple_rnn_9_while_identity_3Msequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity_3:output:0"
Dsequential_18_bidirectional_18_forward_simple_rnn_9_while_identity_4Msequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity_4:output:0"
}sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_strided_slice_1sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_strided_slice_1_0"Þ
lsequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resourcensequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"à
msequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resourceosequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"Ü
ksequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resourcemsequential_18_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0"ú
¹sequential_18_bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor»sequential_18_bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2Ê
csequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpcsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2È
bsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpbsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp2Ì
dsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpdsequential_18/bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
¹
Á
7__inference_backward_simple_rnn_9_layer_call_fn_4706742

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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703733o
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
ß
¯
while_cond_4706797
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4706797___redundant_placeholder05
1while_while_cond_4706797___redundant_placeholder15
1while_while_cond_4706797___redundant_placeholder25
1while_while_cond_4706797___redundant_placeholder3
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
while_body_4703425
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_29_4703447_0:4@0
"while_simple_rnn_cell_29_4703449_0:@4
"while_simple_rnn_cell_29_4703451_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_29_4703447:4@.
 while_simple_rnn_cell_29_4703449:@2
 while_simple_rnn_cell_29_4703451:@@¢0while/simple_rnn_cell_29/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_29/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_29_4703447_0"while_simple_rnn_cell_29_4703449_0"while_simple_rnn_cell_29_4703451_0*
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4703370r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_29/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_29/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_29_4703447"while_simple_rnn_cell_29_4703447_0"F
 while_simple_rnn_cell_29_4703449"while_simple_rnn_cell_29_4703449_0"F
 while_simple_rnn_cell_29_4703451"while_simple_rnn_cell_29_4703451_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_29/StatefulPartitionedCall0while/simple_rnn_cell_29/StatefulPartitionedCall: 
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
5
«
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4703189

inputs,
simple_rnn_cell_28_4703112:4@(
simple_rnn_cell_28_4703114:@,
simple_rnn_cell_28_4703116:@@
identity¢*simple_rnn_cell_28/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_28_4703112simple_rnn_cell_28_4703114simple_rnn_cell_28_4703116*
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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4703072n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_28_4703112simple_rnn_cell_28_4703114simple_rnn_cell_28_4703116*
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
while_body_4703125*
condR
while_cond_4703124*8
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
NoOpNoOp+^simple_rnn_cell_28/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_28/StatefulPartitionedCall*simple_rnn_cell_28/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ß
¯
while_cond_4703424
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4703424___redundant_placeholder05
1while_while_cond_4703424___redundant_placeholder15
1while_while_cond_4703424___redundant_placeholder25
1while_while_cond_4703424___redundant_placeholder3
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
(backward_simple_rnn_9_while_cond_4704518H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2J
Fbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4704518___redundant_placeholder0a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4704518___redundant_placeholder1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4704518___redundant_placeholder2a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4704518___redundant_placeholder3(
$backward_simple_rnn_9_while_identity
º
 backward_simple_rnn_9/while/LessLess'backward_simple_rnn_9_while_placeholderFbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_9/while/IdentityIdentity$backward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0*(
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
Fsequential_18_bidirectional_18_forward_simple_rnn_9_while_cond_4702717
sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_loop_counter
sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_maximum_iterationsI
Esequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholderK
Gsequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholder_1K
Gsequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholder_2
sequential_18_bidirectional_18_forward_simple_rnn_9_while_less_sequential_18_bidirectional_18_forward_simple_rnn_9_strided_slice_1
sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_cond_4702717___redundant_placeholder0
sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_cond_4702717___redundant_placeholder1
sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_cond_4702717___redundant_placeholder2
sequential_18_bidirectional_18_forward_simple_rnn_9_while_sequential_18_bidirectional_18_forward_simple_rnn_9_while_cond_4702717___redundant_placeholder3F
Bsequential_18_bidirectional_18_forward_simple_rnn_9_while_identity
³
>sequential_18/bidirectional_18/forward_simple_rnn_9/while/LessLessEsequential_18_bidirectional_18_forward_simple_rnn_9_while_placeholdersequential_18_bidirectional_18_forward_simple_rnn_9_while_less_sequential_18_bidirectional_18_forward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: ³
Bsequential_18/bidirectional_18/forward_simple_rnn_9/while/IdentityIdentityBsequential_18/bidirectional_18/forward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "
Bsequential_18_bidirectional_18_forward_simple_rnn_9_while_identityKsequential_18/bidirectional_18/forward_simple_rnn_9/while/Identity:output:0*(
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
Ø	
É
%__inference_signature_wrapper_4704761
bidirectional_18_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallbidirectional_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
"__inference__wrapped_model_4702902o
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
_user_specified_namebidirectional_18_input
ß
¯
while_cond_4703124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4703124___redundant_placeholder05
1while_while_cond_4703124___redundant_placeholder15
1while_while_cond_4703124___redundant_placeholder25
1while_while_cond_4703124___redundant_placeholder3
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
äA
É
'forward_simple_rnn_9_while_body_4705368F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2E
Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0
}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_9_while_identity)
%forward_simple_rnn_9_while_identity_1)
%forward_simple_rnn_9_while_identity_2)
%forward_simple_rnn_9_while_identity_3)
%forward_simple_rnn_9_while_identity_4C
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp¢Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
Lforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
>forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_9_while_placeholderUforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ò
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_9/while/simple_rnn_cell_28/MatMulMatMulEforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAddBiasAdd>forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul:product:0Lforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1MatMul(forward_simple_rnn_9_while_placeholder_2Mforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_9/while/simple_rnn_cell_28/addAddV2>forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd:output:0@forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_9/while/simple_rnn_cell_28/TanhTanh5forward_simple_rnn_9/while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_9_while_placeholder_1Nforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_9/while/addAddV2&forward_simple_rnn_9_while_placeholder)forward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_9/while/add_1AddV2Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counter+forward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_9/while/IdentityIdentity$forward_simple_rnn_9/while/add_1:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_9/while/Identity_1IdentityHforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_9/while/Identity_2Identity"forward_simple_rnn_9/while/add:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_9/while/Identity_3IdentityOforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_9/while/Identity_4Identity6forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0 ^forward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_9/while/NoOpNoOpE^forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpD^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpF^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0"S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0"W
%forward_simple_rnn_9_while_identity_1.forward_simple_rnn_9/while/Identity_1:output:0"W
%forward_simple_rnn_9_while_identity_2.forward_simple_rnn_9/while/Identity_2:output:0"W
%forward_simple_rnn_9_while_identity_3.forward_simple_rnn_9/while/Identity_3:output:0"W
%forward_simple_rnn_9_while_identity_4.forward_simple_rnn_9/while/Identity_4:output:0" 
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resourceOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resourcePforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resourceNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpDforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpCforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp2
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpEforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
while_cond_4707133
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4707133___redundant_placeholder05
1while_while_cond_4707133___redundant_placeholder15
1while_while_cond_4707133___redundant_placeholder25
1while_while_cond_4707133___redundant_placeholder3
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
ÿ@
Ë
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703884

inputsC
1simple_rnn_cell_29_matmul_readvariableop_resource:4@@
2simple_rnn_cell_29_biasadd_readvariableop_resource:@E
3simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_29/BiasAdd/ReadVariableOp¢(simple_rnn_cell_29/MatMul/ReadVariableOp¢*simple_rnn_cell_29/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_29/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_29/BiasAddBiasAdd#simple_rnn_cell_29/MatMul:product:01simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_29/MatMul_1MatMulzeros:output:02simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_29/addAddV2#simple_rnn_cell_29/BiasAdd:output:0%simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_29/TanhTanhsimple_rnn_cell_29/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_29_matmul_readvariableop_resource2simple_rnn_cell_29_biasadd_readvariableop_resource3simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
while_body_4703817*
condR
while_cond_4703816*8
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
NoOpNoOp*^simple_rnn_cell_29/BiasAdd/ReadVariableOp)^simple_rnn_cell_29/MatMul/ReadVariableOp+^simple_rnn_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_29/BiasAdd/ReadVariableOp)simple_rnn_cell_29/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_29/MatMul/ReadVariableOp(simple_rnn_cell_29/MatMul/ReadVariableOp2X
*simple_rnn_cell_29/MatMul_1/ReadVariableOp*simple_rnn_cell_29/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
Â
6__inference_forward_simple_rnn_9_layer_call_fn_4706247
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4703189o
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
ß
¯
while_cond_4702963
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4702963___redundant_placeholder05
1while_while_cond_4702963___redundant_placeholder15
1while_while_cond_4702963___redundant_placeholder25
1while_while_cond_4702963___redundant_placeholder3
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

ì
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4707263

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
¹
Á
7__inference_backward_simple_rnn_9_layer_call_fn_4706753

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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703884o
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
ÝB
è
(backward_simple_rnn_9_while_body_4705476H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2G
Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0
backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_9_while_identity*
&backward_simple_rnn_9_while_identity_1*
&backward_simple_rnn_9_while_identity_2*
&backward_simple_rnn_9_while_identity_3*
&backward_simple_rnn_9_while_identity_4E
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp¢Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
?backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_9_while_placeholderVbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ô
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_9/while/simple_rnn_cell_29/MatMulMatMulFbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAddBiasAdd?backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul:product:0Mbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1MatMul)backward_simple_rnn_9_while_placeholder_2Nbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_9/while/simple_rnn_cell_29/addAddV2?backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd:output:0Abackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_9/while/simple_rnn_cell_29/TanhTanh6backward_simple_rnn_9/while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_9_while_placeholder_1Obackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_9/while/addAddV2'backward_simple_rnn_9_while_placeholder*backward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_9/while/add_1AddV2Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counter,backward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_9/while/IdentityIdentity%backward_simple_rnn_9/while/add_1:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_9/while/Identity_1IdentityJbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_9/while/Identity_2Identity#backward_simple_rnn_9/while/add:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_9/while/Identity_3IdentityPbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_9/while/Identity_4Identity7backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0!^backward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_9/while/NoOpNoOpF^backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpE^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpG^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0"U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0"Y
&backward_simple_rnn_9_while_identity_1/backward_simple_rnn_9/while/Identity_1:output:0"Y
&backward_simple_rnn_9_while_identity_2/backward_simple_rnn_9/while/Identity_2:output:0"Y
&backward_simple_rnn_9_while_identity_3/backward_simple_rnn_9/while/Identity_3:output:0"Y
&backward_simple_rnn_9_while_identity_4/backward_simple_rnn_9/while/Identity_4:output:0"¢
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resourcePbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resourceQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resourceObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0"
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpEbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpDbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp2
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpFbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
ÙQ
Ê
8bidirectional_18_forward_simple_rnn_9_while_body_4704846h
dbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_loop_countern
jbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_maximum_iterations;
7bidirectional_18_forward_simple_rnn_9_while_placeholder=
9bidirectional_18_forward_simple_rnn_9_while_placeholder_1=
9bidirectional_18_forward_simple_rnn_9_while_placeholder_2g
cbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_strided_slice_1_0¤
bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0q
_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@n
`bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@s
abidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@8
4bidirectional_18_forward_simple_rnn_9_while_identity:
6bidirectional_18_forward_simple_rnn_9_while_identity_1:
6bidirectional_18_forward_simple_rnn_9_while_identity_2:
6bidirectional_18_forward_simple_rnn_9_while_identity_3:
6bidirectional_18_forward_simple_rnn_9_while_identity_4e
abidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_strided_slice_1¢
bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensoro
]bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource:4@l
^bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource:@q
_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢Ubidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Tbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp¢Vbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp®
]bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   å
Obidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_07bidirectional_18_forward_simple_rnn_9_while_placeholderfbidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ô
Tbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0·
Ebidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMulMatMulVbidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0\bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ò
Ubidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp`bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0³
Fbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAddBiasAddObidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul:product:0]bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ø
Vbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpabidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0
Gbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1MatMul9bidirectional_18_forward_simple_rnn_9_while_placeholder_2^bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
Bbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/addAddV2Obidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd:output:0Qbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
Cbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/TanhTanhFbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Vbidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Pbidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem9bidirectional_18_forward_simple_rnn_9_while_placeholder_1_bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0Gbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒs
1bidirectional_18/forward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Î
/bidirectional_18/forward_simple_rnn_9/while/addAddV27bidirectional_18_forward_simple_rnn_9_while_placeholder:bidirectional_18/forward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: u
3bidirectional_18/forward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ÿ
1bidirectional_18/forward_simple_rnn_9/while/add_1AddV2dbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_loop_counter<bidirectional_18/forward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: Ë
4bidirectional_18/forward_simple_rnn_9/while/IdentityIdentity5bidirectional_18/forward_simple_rnn_9/while/add_1:z:01^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
6bidirectional_18/forward_simple_rnn_9/while/Identity_1Identityjbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_maximum_iterations1^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Ë
6bidirectional_18/forward_simple_rnn_9/while/Identity_2Identity3bidirectional_18/forward_simple_rnn_9/while/add:z:01^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ø
6bidirectional_18/forward_simple_rnn_9/while/Identity_3Identity`bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ð
6bidirectional_18/forward_simple_rnn_9/while/Identity_4IdentityGbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:01^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
0bidirectional_18/forward_simple_rnn_9/while/NoOpNoOpV^bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpU^bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpW^bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "È
abidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_strided_slice_1cbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_strided_slice_1_0"u
4bidirectional_18_forward_simple_rnn_9_while_identity=bidirectional_18/forward_simple_rnn_9/while/Identity:output:0"y
6bidirectional_18_forward_simple_rnn_9_while_identity_1?bidirectional_18/forward_simple_rnn_9/while/Identity_1:output:0"y
6bidirectional_18_forward_simple_rnn_9_while_identity_2?bidirectional_18/forward_simple_rnn_9/while/Identity_2:output:0"y
6bidirectional_18_forward_simple_rnn_9_while_identity_3?bidirectional_18/forward_simple_rnn_9/while/Identity_3:output:0"y
6bidirectional_18_forward_simple_rnn_9_while_identity_4?bidirectional_18/forward_simple_rnn_9/while/Identity_4:output:0"Â
^bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource`bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"Ä
_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resourceabidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"À
]bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0"Â
bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorbidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2®
Ubidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpUbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2¬
Tbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpTbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp2°
Vbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpVbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
while_body_4707134
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_29_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_29/MatMul/ReadVariableOp¢0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_29/BiasAddBiasAdd)while/simple_rnn_cell_29/MatMul:product:07while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_29/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_29/addAddV2)while/simple_rnn_cell_29/BiasAdd:output:0+while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_29/TanhTanh while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_29/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_29/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_29/MatMul/ReadVariableOp1^while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_29_matmul_readvariableop_resource9while_simple_rnn_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_29/MatMul/ReadVariableOp.while/simple_rnn_cell_29/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
4__inference_simple_rnn_cell_28_layer_call_fn_4707229

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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4703072o
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
while_cond_4703948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4703948___redundant_placeholder05
1while_while_cond_4703948___redundant_placeholder15
1while_while_cond_4703948___redundant_placeholder25
1while_while_cond_4703948___redundant_placeholder3
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
ÔB
è
(backward_simple_rnn_9_while_body_4705916H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2G
Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0
backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_9_while_identity*
&backward_simple_rnn_9_while_identity_1*
&backward_simple_rnn_9_while_identity_2*
&backward_simple_rnn_9_while_identity_3*
&backward_simple_rnn_9_while_identity_4E
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp¢Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_9_while_placeholderVbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_9/while/simple_rnn_cell_29/MatMulMatMulFbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAddBiasAdd?backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul:product:0Mbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1MatMul)backward_simple_rnn_9_while_placeholder_2Nbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_9/while/simple_rnn_cell_29/addAddV2?backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd:output:0Abackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_9/while/simple_rnn_cell_29/TanhTanh6backward_simple_rnn_9/while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_9_while_placeholder_1Obackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_9/while/addAddV2'backward_simple_rnn_9_while_placeholder*backward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_9/while/add_1AddV2Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counter,backward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_9/while/IdentityIdentity%backward_simple_rnn_9/while/add_1:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_9/while/Identity_1IdentityJbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_9/while/Identity_2Identity#backward_simple_rnn_9/while/add:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_9/while/Identity_3IdentityPbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_9/while/Identity_4Identity7backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0!^backward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_9/while/NoOpNoOpF^backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpE^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpG^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0"U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0"Y
&backward_simple_rnn_9_while_identity_1/backward_simple_rnn_9/while/Identity_1:output:0"Y
&backward_simple_rnn_9_while_identity_2/backward_simple_rnn_9/while/Identity_2:output:0"Y
&backward_simple_rnn_9_while_identity_3/backward_simple_rnn_9/while/Identity_3:output:0"Y
&backward_simple_rnn_9_while_identity_4/backward_simple_rnn_9/while/Identity_4:output:0"¢
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resourcePbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resourceQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resourceObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0"
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpEbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpDbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp2
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpFbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
while_cond_4706311
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4706311___redundant_placeholder05
1while_while_cond_4706311___redundant_placeholder15
1while_while_cond_4706311___redundant_placeholder25
1while_while_cond_4706311___redundant_placeholder3
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
ÔB
è
(backward_simple_rnn_9_while_body_4704219H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2G
Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0
backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_9_while_identity*
&backward_simple_rnn_9_while_identity_1*
&backward_simple_rnn_9_while_identity_2*
&backward_simple_rnn_9_while_identity_3*
&backward_simple_rnn_9_while_identity_4E
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp¢Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_9_while_placeholderVbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_9/while/simple_rnn_cell_29/MatMulMatMulFbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAddBiasAdd?backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul:product:0Mbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1MatMul)backward_simple_rnn_9_while_placeholder_2Nbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_9/while/simple_rnn_cell_29/addAddV2?backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd:output:0Abackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_9/while/simple_rnn_cell_29/TanhTanh6backward_simple_rnn_9/while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_9_while_placeholder_1Obackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_9/while/addAddV2'backward_simple_rnn_9_while_placeholder*backward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_9/while/add_1AddV2Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counter,backward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_9/while/IdentityIdentity%backward_simple_rnn_9/while/add_1:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_9/while/Identity_1IdentityJbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_9/while/Identity_2Identity#backward_simple_rnn_9/while/add:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_9/while/Identity_3IdentityPbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_9/while/Identity_4Identity7backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0!^backward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_9/while/NoOpNoOpF^backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpE^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpG^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0"U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0"Y
&backward_simple_rnn_9_while_identity_1/backward_simple_rnn_9/while/Identity_1:output:0"Y
&backward_simple_rnn_9_while_identity_2/backward_simple_rnn_9/while/Identity_2:output:0"Y
&backward_simple_rnn_9_while_identity_3/backward_simple_rnn_9/while/Identity_3:output:0"Y
&backward_simple_rnn_9_while_identity_4/backward_simple_rnn_9/while/Identity_4:output:0"¢
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resourcePbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resourceQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resourceObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0"
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpEbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpDbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp2
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpFbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
·	

2__inference_bidirectional_18_layer_call_fn_4705274
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4703744p
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
þ6
¬
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703489

inputs,
simple_rnn_cell_29_4703412:4@(
simple_rnn_cell_29_4703414:@,
simple_rnn_cell_29_4703416:@@
identity¢*simple_rnn_cell_29/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_29_4703412simple_rnn_cell_29_4703414simple_rnn_cell_29_4703416*
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4703370n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_29_4703412simple_rnn_cell_29_4703414simple_rnn_cell_29_4703416*
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
while_body_4703425*
condR
while_cond_4703424*8
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
NoOpNoOp+^simple_rnn_cell_29/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_29/StatefulPartitionedCall*simple_rnn_cell_29/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ÛA
É
'forward_simple_rnn_9_while_body_4706028F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2E
Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0
}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_9_while_identity)
%forward_simple_rnn_9_while_identity_1)
%forward_simple_rnn_9_while_identity_2)
%forward_simple_rnn_9_while_identity_3)
%forward_simple_rnn_9_while_identity_4C
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp¢Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
Lforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
>forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_9_while_placeholderUforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ò
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_9/while/simple_rnn_cell_28/MatMulMatMulEforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAddBiasAdd>forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul:product:0Lforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1MatMul(forward_simple_rnn_9_while_placeholder_2Mforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_9/while/simple_rnn_cell_28/addAddV2>forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd:output:0@forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_9/while/simple_rnn_cell_28/TanhTanh5forward_simple_rnn_9/while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_9_while_placeholder_1Nforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_9/while/addAddV2&forward_simple_rnn_9_while_placeholder)forward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_9/while/add_1AddV2Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counter+forward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_9/while/IdentityIdentity$forward_simple_rnn_9/while/add_1:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_9/while/Identity_1IdentityHforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_9/while/Identity_2Identity"forward_simple_rnn_9/while/add:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_9/while/Identity_3IdentityOforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_9/while/Identity_4Identity6forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0 ^forward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_9/while/NoOpNoOpE^forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpD^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpF^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0"S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0"W
%forward_simple_rnn_9_while_identity_1.forward_simple_rnn_9/while/Identity_1:output:0"W
%forward_simple_rnn_9_while_identity_2.forward_simple_rnn_9/while/Identity_2:output:0"W
%forward_simple_rnn_9_while_identity_3.forward_simple_rnn_9/while/Identity_3:output:0"W
%forward_simple_rnn_9_while_identity_4.forward_simple_rnn_9/while/Identity_4:output:0" 
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resourceOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resourcePforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resourceNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpDforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpCforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp2
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpEforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
'forward_simple_rnn_9_while_body_4704111F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2E
Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0
}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_9_while_identity)
%forward_simple_rnn_9_while_identity_1)
%forward_simple_rnn_9_while_identity_2)
%forward_simple_rnn_9_while_identity_3)
%forward_simple_rnn_9_while_identity_4C
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp¢Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
Lforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
>forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_9_while_placeholderUforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ò
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_9/while/simple_rnn_cell_28/MatMulMatMulEforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAddBiasAdd>forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul:product:0Lforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1MatMul(forward_simple_rnn_9_while_placeholder_2Mforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_9/while/simple_rnn_cell_28/addAddV2>forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd:output:0@forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_9/while/simple_rnn_cell_28/TanhTanh5forward_simple_rnn_9/while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_9_while_placeholder_1Nforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_9/while/addAddV2&forward_simple_rnn_9_while_placeholder)forward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_9/while/add_1AddV2Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counter+forward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_9/while/IdentityIdentity$forward_simple_rnn_9/while/add_1:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_9/while/Identity_1IdentityHforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_9/while/Identity_2Identity"forward_simple_rnn_9/while/add:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_9/while/Identity_3IdentityOforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_9/while/Identity_4Identity6forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0 ^forward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_9/while/NoOpNoOpE^forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpD^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpF^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0"S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0"W
%forward_simple_rnn_9_while_identity_1.forward_simple_rnn_9/while/Identity_1:output:0"W
%forward_simple_rnn_9_while_identity_2.forward_simple_rnn_9/while/Identity_2:output:0"W
%forward_simple_rnn_9_while_identity_3.forward_simple_rnn_9/while/Identity_3:output:0"W
%forward_simple_rnn_9_while_identity_4.forward_simple_rnn_9/while/Identity_4:output:0" 
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resourceOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resourcePforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resourceNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpDforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpCforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp2
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpEforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
ÿ
ê
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4703248

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
while_cond_4706531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4706531___redundant_placeholder05
1while_while_cond_4706531___redundant_placeholder15
1while_while_cond_4706531___redundant_placeholder25
1while_while_cond_4706531___redundant_placeholder3
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706258

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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4703614o
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
§
Í
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704288

inputsX
Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp¢@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp¢backward_simple_rnn_9/while¢>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp¢?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp¢forward_simple_rnn_9/whileP
forward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_9/strided_sliceStridedSlice#forward_simple_rnn_9/Shape:output:01forward_simple_rnn_9/strided_slice/stack:output:03forward_simple_rnn_9/strided_slice/stack_1:output:03forward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_9/zeros/packedPack+forward_simple_rnn_9/strided_slice:output:0,forward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_9/zerosFill*forward_simple_rnn_9/zeros/packed:output:0)forward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_9/transpose	Transposeinputs,forward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4n
forward_simple_rnn_9/Shape_1Shape"forward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_9/strided_slice_1StridedSlice%forward_simple_rnn_9/Shape_1:output:03forward_simple_rnn_9/strided_slice_1/stack:output:05forward_simple_rnn_9/strided_slice_1/stack_1:output:05forward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_9/TensorArrayV2TensorListReserve9forward_simple_rnn_9/TensorArrayV2/element_shape:output:0-forward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
<forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_9/transpose:y:0Sforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$forward_simple_rnn_9/strided_slice_2StridedSlice"forward_simple_rnn_9/transpose:y:03forward_simple_rnn_9/strided_slice_2/stack:output:05forward_simple_rnn_9/strided_slice_2/stack_1:output:05forward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÄ
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_9/simple_rnn_cell_28/MatMulMatMul-forward_simple_rnn_9/strided_slice_2:output:0Eforward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_9/simple_rnn_cell_28/BiasAddBiasAdd8forward_simple_rnn_9/simple_rnn_cell_28/MatMul:product:0Fforward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1MatMul#forward_simple_rnn_9/zeros:output:0Gforward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_9/simple_rnn_cell_28/addAddV28forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd:output:0:forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_9/simple_rnn_cell_28/TanhTanh/forward_simple_rnn_9/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_9/TensorArrayV2_1TensorListReserve;forward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_9/whileWhile0forward_simple_rnn_9/while/loop_counter:output:06forward_simple_rnn_9/while/maximum_iterations:output:0"forward_simple_rnn_9/time:output:0-forward_simple_rnn_9/TensorArrayV2_1:handle:0#forward_simple_rnn_9/zeros:output:0-forward_simple_rnn_9/strided_slice_1:output:0Lforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resourceGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resourceHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
'forward_simple_rnn_9_while_body_4704111*3
cond+R)
'forward_simple_rnn_9_while_cond_4704110*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_9/while:output:3Nforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_9/strided_slice_3StridedSlice@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_9/strided_slice_3/stack:output:05forward_simple_rnn_9/strided_slice_3/stack_1:output:05forward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_9/transpose_1	Transpose@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
backward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_9/strided_sliceStridedSlice$backward_simple_rnn_9/Shape:output:02backward_simple_rnn_9/strided_slice/stack:output:04backward_simple_rnn_9/strided_slice/stack_1:output:04backward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_9/zeros/packedPack,backward_simple_rnn_9/strided_slice:output:0-backward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_9/zerosFill+backward_simple_rnn_9/zeros/packed:output:0*backward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_simple_rnn_9/transpose	Transposeinputs-backward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
backward_simple_rnn_9/Shape_1Shape#backward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_9/strided_slice_1StridedSlice&backward_simple_rnn_9/Shape_1:output:04backward_simple_rnn_9/strided_slice_1/stack:output:06backward_simple_rnn_9/strided_slice_1/stack_1:output:06backward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_9/TensorArrayV2TensorListReserve:backward_simple_rnn_9/TensorArrayV2/element_shape:output:0.backward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_9/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¶
backward_simple_rnn_9/ReverseV2	ReverseV2#backward_simple_rnn_9/transpose:y:0-backward_simple_rnn_9/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Kbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   §
=backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_9/ReverseV2:output:0Tbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%backward_simple_rnn_9/strided_slice_2StridedSlice#backward_simple_rnn_9/transpose:y:04backward_simple_rnn_9/strided_slice_2/stack:output:06backward_simple_rnn_9/strided_slice_2/stack_1:output:06backward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_9/simple_rnn_cell_29/MatMulMatMul.backward_simple_rnn_9/strided_slice_2:output:0Fbackward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_9/simple_rnn_cell_29/BiasAddBiasAdd9backward_simple_rnn_9/simple_rnn_cell_29/MatMul:product:0Gbackward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1MatMul$backward_simple_rnn_9/zeros:output:0Hbackward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_9/simple_rnn_cell_29/addAddV29backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd:output:0;backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_9/simple_rnn_cell_29/TanhTanh0backward_simple_rnn_9/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_9/TensorArrayV2_1TensorListReserve<backward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_9/whileWhile1backward_simple_rnn_9/while/loop_counter:output:07backward_simple_rnn_9/while/maximum_iterations:output:0#backward_simple_rnn_9/time:output:0.backward_simple_rnn_9/TensorArrayV2_1:handle:0$backward_simple_rnn_9/zeros:output:0.backward_simple_rnn_9/strided_slice_1:output:0Mbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resourceHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resourceIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
(backward_simple_rnn_9_while_body_4704219*4
cond,R*
(backward_simple_rnn_9_while_cond_4704218*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_9/while:output:3Obackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_9/strided_slice_3StridedSliceAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_9/strided_slice_3/stack:output:06backward_simple_rnn_9/strided_slice_3/stack_1:output:06backward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_9/transpose_1	TransposeAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_9/strided_slice_3:output:0.backward_simple_rnn_9/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?^backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpA^backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp^backward_simple_rnn_9/while?^forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>^forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp@^forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp^forward_simple_rnn_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp2
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp2:
backward_simple_rnn_9/whilebackward_simple_rnn_9/while2
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp2
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp28
forward_simple_rnn_9/whileforward_simple_rnn_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
²
Ñ
(backward_simple_rnn_9_while_cond_4704218H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2J
Fbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4704218___redundant_placeholder0a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4704218___redundant_placeholder1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4704218___redundant_placeholder2a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4704218___redundant_placeholder3(
$backward_simple_rnn_9_while_identity
º
 backward_simple_rnn_9/while/LessLess'backward_simple_rnn_9_while_placeholderFbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_9/while/IdentityIdentity$backward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0*(
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705765
inputs_0X
Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp¢@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp¢backward_simple_rnn_9/while¢>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp¢?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp¢forward_simple_rnn_9/whileR
forward_simple_rnn_9/ShapeShapeinputs_0*
T0*
_output_shapes
:r
(forward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_9/strided_sliceStridedSlice#forward_simple_rnn_9/Shape:output:01forward_simple_rnn_9/strided_slice/stack:output:03forward_simple_rnn_9/strided_slice/stack_1:output:03forward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_9/zeros/packedPack+forward_simple_rnn_9/strided_slice:output:0,forward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_9/zerosFill*forward_simple_rnn_9/zeros/packed:output:0)forward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
forward_simple_rnn_9/transpose	Transposeinputs_0,forward_simple_rnn_9/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn
forward_simple_rnn_9/Shape_1Shape"forward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_9/strided_slice_1StridedSlice%forward_simple_rnn_9/Shape_1:output:03forward_simple_rnn_9/strided_slice_1/stack:output:05forward_simple_rnn_9/strided_slice_1/stack_1:output:05forward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_9/TensorArrayV2TensorListReserve9forward_simple_rnn_9/TensorArrayV2/element_shape:output:0-forward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
<forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_9/transpose:y:0Sforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
$forward_simple_rnn_9/strided_slice_2StridedSlice"forward_simple_rnn_9/transpose:y:03forward_simple_rnn_9/strided_slice_2/stack:output:05forward_simple_rnn_9/strided_slice_2/stack_1:output:05forward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÄ
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_9/simple_rnn_cell_28/MatMulMatMul-forward_simple_rnn_9/strided_slice_2:output:0Eforward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_9/simple_rnn_cell_28/BiasAddBiasAdd8forward_simple_rnn_9/simple_rnn_cell_28/MatMul:product:0Fforward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1MatMul#forward_simple_rnn_9/zeros:output:0Gforward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_9/simple_rnn_cell_28/addAddV28forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd:output:0:forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_9/simple_rnn_cell_28/TanhTanh/forward_simple_rnn_9/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_9/TensorArrayV2_1TensorListReserve;forward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_9/whileWhile0forward_simple_rnn_9/while/loop_counter:output:06forward_simple_rnn_9/while/maximum_iterations:output:0"forward_simple_rnn_9/time:output:0-forward_simple_rnn_9/TensorArrayV2_1:handle:0#forward_simple_rnn_9/zeros:output:0-forward_simple_rnn_9/strided_slice_1:output:0Lforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resourceGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resourceHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
'forward_simple_rnn_9_while_body_4705588*3
cond+R)
'forward_simple_rnn_9_while_cond_4705587*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_9/while:output:3Nforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_9/strided_slice_3StridedSlice@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_9/strided_slice_3/stack:output:05forward_simple_rnn_9/strided_slice_3/stack_1:output:05forward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_9/transpose_1	Transpose@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
backward_simple_rnn_9/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)backward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_9/strided_sliceStridedSlice$backward_simple_rnn_9/Shape:output:02backward_simple_rnn_9/strided_slice/stack:output:04backward_simple_rnn_9/strided_slice/stack_1:output:04backward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_9/zeros/packedPack,backward_simple_rnn_9/strided_slice:output:0-backward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_9/zerosFill+backward_simple_rnn_9/zeros/packed:output:0*backward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
backward_simple_rnn_9/transpose	Transposeinputs_0-backward_simple_rnn_9/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
backward_simple_rnn_9/Shape_1Shape#backward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_9/strided_slice_1StridedSlice&backward_simple_rnn_9/Shape_1:output:04backward_simple_rnn_9/strided_slice_1/stack:output:06backward_simple_rnn_9/strided_slice_1/stack_1:output:06backward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_9/TensorArrayV2TensorListReserve:backward_simple_rnn_9/TensorArrayV2/element_shape:output:0.backward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_9/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: È
backward_simple_rnn_9/ReverseV2	ReverseV2#backward_simple_rnn_9/transpose:y:0-backward_simple_rnn_9/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Kbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ§
=backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_9/ReverseV2:output:0Tbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
%backward_simple_rnn_9/strided_slice_2StridedSlice#backward_simple_rnn_9/transpose:y:04backward_simple_rnn_9/strided_slice_2/stack:output:06backward_simple_rnn_9/strided_slice_2/stack_1:output:06backward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_9/simple_rnn_cell_29/MatMulMatMul.backward_simple_rnn_9/strided_slice_2:output:0Fbackward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_9/simple_rnn_cell_29/BiasAddBiasAdd9backward_simple_rnn_9/simple_rnn_cell_29/MatMul:product:0Gbackward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1MatMul$backward_simple_rnn_9/zeros:output:0Hbackward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_9/simple_rnn_cell_29/addAddV29backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd:output:0;backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_9/simple_rnn_cell_29/TanhTanh0backward_simple_rnn_9/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_9/TensorArrayV2_1TensorListReserve<backward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_9/whileWhile1backward_simple_rnn_9/while/loop_counter:output:07backward_simple_rnn_9/while/maximum_iterations:output:0#backward_simple_rnn_9/time:output:0.backward_simple_rnn_9/TensorArrayV2_1:handle:0$backward_simple_rnn_9/zeros:output:0.backward_simple_rnn_9/strided_slice_1:output:0Mbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resourceHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resourceIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
(backward_simple_rnn_9_while_body_4705696*4
cond,R*
(backward_simple_rnn_9_while_cond_4705695*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_9/while:output:3Obackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_9/strided_slice_3StridedSliceAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_9/strided_slice_3/stack:output:06backward_simple_rnn_9/strided_slice_3/stack_1:output:06backward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_9/transpose_1	TransposeAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_9/strided_slice_3:output:0.backward_simple_rnn_9/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?^backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpA^backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp^backward_simple_rnn_9/while?^forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>^forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp@^forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp^forward_simple_rnn_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp2
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp2:
backward_simple_rnn_9/whilebackward_simple_rnn_9/while2
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp2
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp28
forward_simple_rnn_9/whileforward_simple_rnn_9/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ß
¯
while_cond_4706641
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4706641___redundant_placeholder05
1while_while_cond_4706641___redundant_placeholder15
1while_while_cond_4706641___redundant_placeholder25
1while_while_cond_4706641___redundant_placeholder3
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
õ_

Gsequential_18_bidirectional_18_backward_simple_rnn_9_while_body_4702826
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_loop_counter
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_maximum_iterationsJ
Fsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholderL
Hsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholder_1L
Hsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholder_2
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_strided_slice_1_0Â
½sequential_18_bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0
nsequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@}
osequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@
psequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@G
Csequential_18_bidirectional_18_backward_simple_rnn_9_while_identityI
Esequential_18_bidirectional_18_backward_simple_rnn_9_while_identity_1I
Esequential_18_bidirectional_18_backward_simple_rnn_9_while_identity_2I
Esequential_18_bidirectional_18_backward_simple_rnn_9_while_identity_3I
Esequential_18_bidirectional_18_backward_simple_rnn_9_while_identity_4
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_strided_slice_1À
»sequential_18_bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor~
lsequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource:4@{
msequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource:@
nsequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢dsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢csequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp¢esequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp½
lsequential_18/bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   °
^sequential_18/bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem½sequential_18_bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0Fsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholderusequential_18/bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0
csequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpnsequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0ä
Tsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMulMatMulesequential_18/bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0ksequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOposequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0à
Usequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAddBiasAdd^sequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul:product:0lsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
esequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOppsequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Ë
Vsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1MatMulHsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholder_2msequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
Qsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/addAddV2^sequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd:output:0`sequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ã
Rsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/TanhTanhUsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
esequential_18/bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
_sequential_18/bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemHsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholder_1nsequential_18/bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0Vsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ
@sequential_18/bidirectional_18/backward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :û
>sequential_18/bidirectional_18/backward_simple_rnn_9/while/addAddV2Fsequential_18_bidirectional_18_backward_simple_rnn_9_while_placeholderIsequential_18/bidirectional_18/backward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: 
Bsequential_18/bidirectional_18/backward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¼
@sequential_18/bidirectional_18/backward_simple_rnn_9/while/add_1AddV2sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_loop_counterKsequential_18/bidirectional_18/backward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: ø
Csequential_18/bidirectional_18/backward_simple_rnn_9/while/IdentityIdentityDsequential_18/bidirectional_18/backward_simple_rnn_9/while/add_1:z:0@^sequential_18/bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ¿
Esequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity_1Identitysequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_while_maximum_iterations@^sequential_18/bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ø
Esequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity_2IdentityBsequential_18/bidirectional_18/backward_simple_rnn_9/while/add:z:0@^sequential_18/bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ¥
Esequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity_3Identityosequential_18/bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0@^sequential_18/bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
Esequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity_4IdentityVsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0@^sequential_18/bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
?sequential_18/bidirectional_18/backward_simple_rnn_9/while/NoOpNoOpe^sequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpd^sequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpf^sequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Csequential_18_bidirectional_18_backward_simple_rnn_9_while_identityLsequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity:output:0"
Esequential_18_bidirectional_18_backward_simple_rnn_9_while_identity_1Nsequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity_1:output:0"
Esequential_18_bidirectional_18_backward_simple_rnn_9_while_identity_2Nsequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity_2:output:0"
Esequential_18_bidirectional_18_backward_simple_rnn_9_while_identity_3Nsequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity_3:output:0"
Esequential_18_bidirectional_18_backward_simple_rnn_9_while_identity_4Nsequential_18/bidirectional_18/backward_simple_rnn_9/while/Identity_4:output:0"
sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_strided_slice_1sequential_18_bidirectional_18_backward_simple_rnn_9_while_sequential_18_bidirectional_18_backward_simple_rnn_9_strided_slice_1_0"à
msequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resourceosequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"â
nsequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resourcepsequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0"Þ
lsequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resourcensequential_18_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0"þ
»sequential_18_bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor½sequential_18_bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_18_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2Ì
dsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpdsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2Ê
csequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpcsequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp2Î
esequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpesequential_18/bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
E__inference_dense_18_layer_call_and_return_conditional_losses_4704313

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
ÛA
É
'forward_simple_rnn_9_while_body_4704411F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2E
Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0
}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_9_while_identity)
%forward_simple_rnn_9_while_identity_1)
%forward_simple_rnn_9_while_identity_2)
%forward_simple_rnn_9_while_identity_3)
%forward_simple_rnn_9_while_identity_4C
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp¢Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
Lforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
>forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_9_while_placeholderUforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ò
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_9/while/simple_rnn_cell_28/MatMulMatMulEforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAddBiasAdd>forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul:product:0Lforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1MatMul(forward_simple_rnn_9_while_placeholder_2Mforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_9/while/simple_rnn_cell_28/addAddV2>forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd:output:0@forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_9/while/simple_rnn_cell_28/TanhTanh5forward_simple_rnn_9/while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_9_while_placeholder_1Nforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_9/while/addAddV2&forward_simple_rnn_9_while_placeholder)forward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_9/while/add_1AddV2Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counter+forward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_9/while/IdentityIdentity$forward_simple_rnn_9/while/add_1:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_9/while/Identity_1IdentityHforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_9/while/Identity_2Identity"forward_simple_rnn_9/while/add:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_9/while/Identity_3IdentityOforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_9/while/Identity_4Identity6forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0 ^forward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_9/while/NoOpNoOpE^forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpD^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpF^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0"S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0"W
%forward_simple_rnn_9_while_identity_1.forward_simple_rnn_9/while/Identity_1:output:0"W
%forward_simple_rnn_9_while_identity_2.forward_simple_rnn_9/while/Identity_2:output:0"W
%forward_simple_rnn_9_while_identity_3.forward_simple_rnn_9/while/Identity_3:output:0"W
%forward_simple_rnn_9_while_identity_4.forward_simple_rnn_9/while/Identity_4:output:0" 
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resourceOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resourcePforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resourceNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpDforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpCforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp2
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpEforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
	
8bidirectional_18_forward_simple_rnn_9_while_cond_4704845h
dbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_loop_countern
jbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_maximum_iterations;
7bidirectional_18_forward_simple_rnn_9_while_placeholder=
9bidirectional_18_forward_simple_rnn_9_while_placeholder_1=
9bidirectional_18_forward_simple_rnn_9_while_placeholder_2j
fbidirectional_18_forward_simple_rnn_9_while_less_bidirectional_18_forward_simple_rnn_9_strided_slice_1
}bidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_cond_4704845___redundant_placeholder0
}bidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_cond_4704845___redundant_placeholder1
}bidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_cond_4704845___redundant_placeholder2
}bidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_cond_4704845___redundant_placeholder38
4bidirectional_18_forward_simple_rnn_9_while_identity
ú
0bidirectional_18/forward_simple_rnn_9/while/LessLess7bidirectional_18_forward_simple_rnn_9_while_placeholderfbidirectional_18_forward_simple_rnn_9_while_less_bidirectional_18_forward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: 
4bidirectional_18/forward_simple_rnn_9/while/IdentityIdentity4bidirectional_18/forward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "u
4bidirectional_18_forward_simple_rnn_9_while_identity=bidirectional_18/forward_simple_rnn_9/while/Identity:output:0*(
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
ä

J__inference_sequential_18_layer_call_and_return_conditional_losses_4704732
bidirectional_18_input*
bidirectional_18_4704713:4@&
bidirectional_18_4704715:@*
bidirectional_18_4704717:@@*
bidirectional_18_4704719:4@&
bidirectional_18_4704721:@*
bidirectional_18_4704723:@@#
dense_18_4704726:	
dense_18_4704728:
identity¢(bidirectional_18/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall
(bidirectional_18/StatefulPartitionedCallStatefulPartitionedCallbidirectional_18_inputbidirectional_18_4704713bidirectional_18_4704715bidirectional_18_4704717bidirectional_18_4704719bidirectional_18_4704721bidirectional_18_4704723*
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704588¡
 dense_18/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_18/StatefulPartitionedCall:output:0dense_18_4704726dense_18_4704728*
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
E__inference_dense_18_layer_call_and_return_conditional_losses_4704313x
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_18/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_18/StatefulPartitionedCall(bidirectional_18/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_18_input
á

#__inference__traced_restore_4707556
file_prefix3
 assignvariableop_dense_18_kernel:	.
 assignvariableop_1_dense_18_bias:d
Rassignvariableop_2_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel:4@n
\assignvariableop_3_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel:@@^
Passignvariableop_4_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias:@e
Sassignvariableop_5_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel:4@o
]assignvariableop_6_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel:@@_
Qassignvariableop_7_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias:@&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: =
*assignvariableop_17_adam_dense_18_kernel_m:	6
(assignvariableop_18_adam_dense_18_bias_m:l
Zassignvariableop_19_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_m:4@v
dassignvariableop_20_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_m:@@f
Xassignvariableop_21_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_m:@m
[assignvariableop_22_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_m:4@w
eassignvariableop_23_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_m:@@g
Yassignvariableop_24_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_m:@=
*assignvariableop_25_adam_dense_18_kernel_v:	6
(assignvariableop_26_adam_dense_18_bias_v:l
Zassignvariableop_27_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_v:4@v
dassignvariableop_28_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_v:@@f
Xassignvariableop_29_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_v:@m
[assignvariableop_30_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_v:4@w
eassignvariableop_31_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_v:@@g
Yassignvariableop_32_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_v:@
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
AssignVariableOpAssignVariableOp assignvariableop_dense_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_2AssignVariableOpRassignvariableop_2_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_3AssignVariableOp\assignvariableop_3_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_4AssignVariableOpPassignvariableop_4_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_5AssignVariableOpSassignvariableop_5_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_6AssignVariableOp]assignvariableop_6_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_7AssignVariableOpQassignvariableop_7_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_18_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_18_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_19AssignVariableOpZassignvariableop_19_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_20AssignVariableOpdassignvariableop_20_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_21AssignVariableOpXassignvariableop_21_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_22AssignVariableOp[assignvariableop_22_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_23AssignVariableOpeassignvariableop_23_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_24AssignVariableOpYassignvariableop_24_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_18_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_18_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_27AssignVariableOpZassignvariableop_27_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_28AssignVariableOpdassignvariableop_28_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_29AssignVariableOpXassignvariableop_29_adam_bidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_30AssignVariableOp[assignvariableop_30_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_31AssignVariableOpeassignvariableop_31_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_32AssignVariableOpYassignvariableop_32_adam_bidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_bias_vIdentity_32:output:0"/device:CPU:0*
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
û§
Ï
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705545
inputs_0X
Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp¢@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp¢backward_simple_rnn_9/while¢>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp¢?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp¢forward_simple_rnn_9/whileR
forward_simple_rnn_9/ShapeShapeinputs_0*
T0*
_output_shapes
:r
(forward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_9/strided_sliceStridedSlice#forward_simple_rnn_9/Shape:output:01forward_simple_rnn_9/strided_slice/stack:output:03forward_simple_rnn_9/strided_slice/stack_1:output:03forward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_9/zeros/packedPack+forward_simple_rnn_9/strided_slice:output:0,forward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_9/zerosFill*forward_simple_rnn_9/zeros/packed:output:0)forward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
forward_simple_rnn_9/transpose	Transposeinputs_0,forward_simple_rnn_9/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn
forward_simple_rnn_9/Shape_1Shape"forward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_9/strided_slice_1StridedSlice%forward_simple_rnn_9/Shape_1:output:03forward_simple_rnn_9/strided_slice_1/stack:output:05forward_simple_rnn_9/strided_slice_1/stack_1:output:05forward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_9/TensorArrayV2TensorListReserve9forward_simple_rnn_9/TensorArrayV2/element_shape:output:0-forward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
<forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_9/transpose:y:0Sforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
$forward_simple_rnn_9/strided_slice_2StridedSlice"forward_simple_rnn_9/transpose:y:03forward_simple_rnn_9/strided_slice_2/stack:output:05forward_simple_rnn_9/strided_slice_2/stack_1:output:05forward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÄ
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_9/simple_rnn_cell_28/MatMulMatMul-forward_simple_rnn_9/strided_slice_2:output:0Eforward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_9/simple_rnn_cell_28/BiasAddBiasAdd8forward_simple_rnn_9/simple_rnn_cell_28/MatMul:product:0Fforward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1MatMul#forward_simple_rnn_9/zeros:output:0Gforward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_9/simple_rnn_cell_28/addAddV28forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd:output:0:forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_9/simple_rnn_cell_28/TanhTanh/forward_simple_rnn_9/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_9/TensorArrayV2_1TensorListReserve;forward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_9/whileWhile0forward_simple_rnn_9/while/loop_counter:output:06forward_simple_rnn_9/while/maximum_iterations:output:0"forward_simple_rnn_9/time:output:0-forward_simple_rnn_9/TensorArrayV2_1:handle:0#forward_simple_rnn_9/zeros:output:0-forward_simple_rnn_9/strided_slice_1:output:0Lforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resourceGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resourceHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
'forward_simple_rnn_9_while_body_4705368*3
cond+R)
'forward_simple_rnn_9_while_cond_4705367*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_9/while:output:3Nforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_9/strided_slice_3StridedSlice@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_9/strided_slice_3/stack:output:05forward_simple_rnn_9/strided_slice_3/stack_1:output:05forward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_9/transpose_1	Transpose@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
backward_simple_rnn_9/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)backward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_9/strided_sliceStridedSlice$backward_simple_rnn_9/Shape:output:02backward_simple_rnn_9/strided_slice/stack:output:04backward_simple_rnn_9/strided_slice/stack_1:output:04backward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_9/zeros/packedPack,backward_simple_rnn_9/strided_slice:output:0-backward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_9/zerosFill+backward_simple_rnn_9/zeros/packed:output:0*backward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
backward_simple_rnn_9/transpose	Transposeinputs_0-backward_simple_rnn_9/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
backward_simple_rnn_9/Shape_1Shape#backward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_9/strided_slice_1StridedSlice&backward_simple_rnn_9/Shape_1:output:04backward_simple_rnn_9/strided_slice_1/stack:output:06backward_simple_rnn_9/strided_slice_1/stack_1:output:06backward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_9/TensorArrayV2TensorListReserve:backward_simple_rnn_9/TensorArrayV2/element_shape:output:0.backward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_9/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: È
backward_simple_rnn_9/ReverseV2	ReverseV2#backward_simple_rnn_9/transpose:y:0-backward_simple_rnn_9/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Kbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ§
=backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_9/ReverseV2:output:0Tbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
%backward_simple_rnn_9/strided_slice_2StridedSlice#backward_simple_rnn_9/transpose:y:04backward_simple_rnn_9/strided_slice_2/stack:output:06backward_simple_rnn_9/strided_slice_2/stack_1:output:06backward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_9/simple_rnn_cell_29/MatMulMatMul.backward_simple_rnn_9/strided_slice_2:output:0Fbackward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_9/simple_rnn_cell_29/BiasAddBiasAdd9backward_simple_rnn_9/simple_rnn_cell_29/MatMul:product:0Gbackward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1MatMul$backward_simple_rnn_9/zeros:output:0Hbackward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_9/simple_rnn_cell_29/addAddV29backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd:output:0;backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_9/simple_rnn_cell_29/TanhTanh0backward_simple_rnn_9/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_9/TensorArrayV2_1TensorListReserve<backward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_9/whileWhile1backward_simple_rnn_9/while/loop_counter:output:07backward_simple_rnn_9/while/maximum_iterations:output:0#backward_simple_rnn_9/time:output:0.backward_simple_rnn_9/TensorArrayV2_1:handle:0$backward_simple_rnn_9/zeros:output:0.backward_simple_rnn_9/strided_slice_1:output:0Mbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resourceHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resourceIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
(backward_simple_rnn_9_while_body_4705476*4
cond,R*
(backward_simple_rnn_9_while_cond_4705475*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_9/while:output:3Obackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_9/strided_slice_3StridedSliceAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_9/strided_slice_3/stack:output:06backward_simple_rnn_9/strided_slice_3/stack_1:output:06backward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_9/transpose_1	TransposeAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_9/strided_slice_3:output:0.backward_simple_rnn_9/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?^backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpA^backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp^backward_simple_rnn_9/while?^forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>^forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp@^forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp^forward_simple_rnn_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp2
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp2:
backward_simple_rnn_9/whilebackward_simple_rnn_9/while2
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp2
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp28
forward_simple_rnn_9/whileforward_simple_rnn_9/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ó-
Ò
while_body_4706798
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_29_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_29/MatMul/ReadVariableOp¢0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_29/BiasAddBiasAdd)while/simple_rnn_cell_29/MatMul:product:07while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_29/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_29/addAddV2)while/simple_rnn_cell_29/BiasAdd:output:0+while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_29/TanhTanh while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_29/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_29/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_29/MatMul/ReadVariableOp1^while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_29_matmul_readvariableop_resource9while_simple_rnn_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_29/MatMul/ReadVariableOp.while/simple_rnn_cell_29/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4705030

inputsi
Wbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource:4@f
Xbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource:@k
Ybidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@j
Xbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource:4@g
Ybidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource:@l
Zbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@:
'dense_18_matmul_readvariableop_resource:	6
(dense_18_biasadd_readvariableop_resource:
identity¢Pbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Obidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp¢Qbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp¢,bidirectional_18/backward_simple_rnn_9/while¢Obidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Nbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp¢Pbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp¢+bidirectional_18/forward_simple_rnn_9/while¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOpa
+bidirectional_18/forward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:
9bidirectional_18/forward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;bidirectional_18/forward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;bidirectional_18/forward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3bidirectional_18/forward_simple_rnn_9/strided_sliceStridedSlice4bidirectional_18/forward_simple_rnn_9/Shape:output:0Bbidirectional_18/forward_simple_rnn_9/strided_slice/stack:output:0Dbidirectional_18/forward_simple_rnn_9/strided_slice/stack_1:output:0Dbidirectional_18/forward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4bidirectional_18/forward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@å
2bidirectional_18/forward_simple_rnn_9/zeros/packedPack<bidirectional_18/forward_simple_rnn_9/strided_slice:output:0=bidirectional_18/forward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:v
1bidirectional_18/forward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Þ
+bidirectional_18/forward_simple_rnn_9/zerosFill;bidirectional_18/forward_simple_rnn_9/zeros/packed:output:0:bidirectional_18/forward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4bidirectional_18/forward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¹
/bidirectional_18/forward_simple_rnn_9/transpose	Transposeinputs=bidirectional_18/forward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
-bidirectional_18/forward_simple_rnn_9/Shape_1Shape3bidirectional_18/forward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:
;bidirectional_18/forward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_18/forward_simple_rnn_9/strided_slice_1StridedSlice6bidirectional_18/forward_simple_rnn_9/Shape_1:output:0Dbidirectional_18/forward_simple_rnn_9/strided_slice_1/stack:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_1:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Abidirectional_18/forward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
3bidirectional_18/forward_simple_rnn_9/TensorArrayV2TensorListReserveJbidirectional_18/forward_simple_rnn_9/TensorArrayV2/element_shape:output:0>bidirectional_18/forward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¬
[bidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ò
Mbidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor3bidirectional_18/forward_simple_rnn_9/transpose:y:0dbidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;bidirectional_18/forward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:§
5bidirectional_18/forward_simple_rnn_9/strided_slice_2StridedSlice3bidirectional_18/forward_simple_rnn_9/transpose:y:0Dbidirectional_18/forward_simple_rnn_9/strided_slice_2/stack:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_1:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskæ
Nbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpWbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
?bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMulMatMul>bidirectional_18/forward_simple_rnn_9/strided_slice_2:output:0Vbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Obidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpXbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¡
@bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAddBiasAddIbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul:product:0Wbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
Pbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpYbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Abidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1MatMul4bidirectional_18/forward_simple_rnn_9/zeros:output:0Xbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
<bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/addAddV2Ibidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd:output:0Kbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¹
=bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/TanhTanh@bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Cbidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Bbidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :·
5bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1TensorListReserveLbidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0Kbidirectional_18/forward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
*bidirectional_18/forward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 
>bidirectional_18/forward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿz
8bidirectional_18/forward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ë	
+bidirectional_18/forward_simple_rnn_9/whileWhileAbidirectional_18/forward_simple_rnn_9/while/loop_counter:output:0Gbidirectional_18/forward_simple_rnn_9/while/maximum_iterations:output:03bidirectional_18/forward_simple_rnn_9/time:output:0>bidirectional_18/forward_simple_rnn_9/TensorArrayV2_1:handle:04bidirectional_18/forward_simple_rnn_9/zeros:output:0>bidirectional_18/forward_simple_rnn_9/strided_slice_1:output:0]bidirectional_18/forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Wbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resourceXbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resourceYbidirectional_18_forward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
8bidirectional_18_forward_simple_rnn_9_while_body_4704846*D
cond<R:
8bidirectional_18_forward_simple_rnn_9_while_cond_4704845*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations §
Vbidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   È
Hbidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack4bidirectional_18/forward_simple_rnn_9/while:output:3_bidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
;bidirectional_18/forward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
=bidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Å
5bidirectional_18/forward_simple_rnn_9/strided_slice_3StridedSliceQbidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0Dbidirectional_18/forward_simple_rnn_9/strided_slice_3/stack:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_1:output:0Fbidirectional_18/forward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
6bidirectional_18/forward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
1bidirectional_18/forward_simple_rnn_9/transpose_1	TransposeQbidirectional_18/forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0?bidirectional_18/forward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
,bidirectional_18/backward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:
:bidirectional_18/backward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<bidirectional_18/backward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<bidirectional_18/backward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4bidirectional_18/backward_simple_rnn_9/strided_sliceStridedSlice5bidirectional_18/backward_simple_rnn_9/Shape:output:0Cbidirectional_18/backward_simple_rnn_9/strided_slice/stack:output:0Ebidirectional_18/backward_simple_rnn_9/strided_slice/stack_1:output:0Ebidirectional_18/backward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5bidirectional_18/backward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@è
3bidirectional_18/backward_simple_rnn_9/zeros/packedPack=bidirectional_18/backward_simple_rnn_9/strided_slice:output:0>bidirectional_18/backward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:w
2bidirectional_18/backward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    á
,bidirectional_18/backward_simple_rnn_9/zerosFill<bidirectional_18/backward_simple_rnn_9/zeros/packed:output:0;bidirectional_18/backward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5bidirectional_18/backward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
0bidirectional_18/backward_simple_rnn_9/transpose	Transposeinputs>bidirectional_18/backward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
.bidirectional_18/backward_simple_rnn_9/Shape_1Shape4bidirectional_18/backward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:
<bidirectional_18/backward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_18/backward_simple_rnn_9/strided_slice_1StridedSlice7bidirectional_18/backward_simple_rnn_9/Shape_1:output:0Ebidirectional_18/backward_simple_rnn_9/strided_slice_1/stack:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_1:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bbidirectional_18/backward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
4bidirectional_18/backward_simple_rnn_9/TensorArrayV2TensorListReserveKbidirectional_18/backward_simple_rnn_9/TensorArrayV2/element_shape:output:0?bidirectional_18/backward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5bidirectional_18/backward_simple_rnn_9/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: é
0bidirectional_18/backward_simple_rnn_9/ReverseV2	ReverseV24bidirectional_18/backward_simple_rnn_9/transpose:y:0>bidirectional_18/backward_simple_rnn_9/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4­
\bidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ú
Nbidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor9bidirectional_18/backward_simple_rnn_9/ReverseV2:output:0ebidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<bidirectional_18/backward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
6bidirectional_18/backward_simple_rnn_9/strided_slice_2StridedSlice4bidirectional_18/backward_simple_rnn_9/transpose:y:0Ebidirectional_18/backward_simple_rnn_9/strided_slice_2/stack:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_1:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskè
Obidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpXbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
@bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMulMatMul?bidirectional_18/backward_simple_rnn_9/strided_slice_2:output:0Wbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Pbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpYbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Abidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAddBiasAddJbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul:product:0Xbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ì
Qbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpZbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Bbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1MatMul5bidirectional_18/backward_simple_rnn_9/zeros:output:0Ybidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
=bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/addAddV2Jbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd:output:0Lbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
>bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/TanhTanhAbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dbidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Cbidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :º
6bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1TensorListReserveMbidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0Lbidirectional_18/backward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
+bidirectional_18/backward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 
?bidirectional_18/backward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ{
9bidirectional_18/backward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø	
,bidirectional_18/backward_simple_rnn_9/whileWhileBbidirectional_18/backward_simple_rnn_9/while/loop_counter:output:0Hbidirectional_18/backward_simple_rnn_9/while/maximum_iterations:output:04bidirectional_18/backward_simple_rnn_9/time:output:0?bidirectional_18/backward_simple_rnn_9/TensorArrayV2_1:handle:05bidirectional_18/backward_simple_rnn_9/zeros:output:0?bidirectional_18/backward_simple_rnn_9/strided_slice_1:output:0^bidirectional_18/backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Xbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resourceYbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resourceZbidirectional_18_backward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
9bidirectional_18_backward_simple_rnn_9_while_body_4704954*E
cond=R;
9bidirectional_18_backward_simple_rnn_9_while_cond_4704953*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¨
Wbidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
Ibidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack5bidirectional_18/backward_simple_rnn_9/while:output:3`bidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
<bidirectional_18/backward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
>bidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ê
6bidirectional_18/backward_simple_rnn_9/strided_slice_3StridedSliceRbidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0Ebidirectional_18/backward_simple_rnn_9/strided_slice_3/stack:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_1:output:0Gbidirectional_18/backward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
7bidirectional_18/backward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
2bidirectional_18/backward_simple_rnn_9/transpose_1	TransposeRbidirectional_18/backward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional_18/backward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
bidirectional_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
bidirectional_18/concatConcatV2>bidirectional_18/forward_simple_rnn_9/strided_slice_3:output:0?bidirectional_18/backward_simple_rnn_9/strided_slice_3:output:0%bidirectional_18/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_18/MatMulMatMul bidirectional_18/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_18/SoftmaxSoftmaxdense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_18/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
NoOpNoOpQ^bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpP^bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpR^bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp-^bidirectional_18/backward_simple_rnn_9/whileP^bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpO^bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpQ^bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp,^bidirectional_18/forward_simple_rnn_9/while ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2¤
Pbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpPbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp2¢
Obidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpObidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp2¦
Qbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpQbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp2\
,bidirectional_18/backward_simple_rnn_9/while,bidirectional_18/backward_simple_rnn_9/while2¢
Obidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpObidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp2 
Nbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpNbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp2¤
Pbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpPbidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp2Z
+bidirectional_18/forward_simple_rnn_9/while+bidirectional_18/forward_simple_rnn_9/while2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs


Ó
/__inference_sequential_18_layer_call_fn_4704688
bidirectional_18_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704648o
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
_user_specified_namebidirectional_18_input
ß
¯
while_cond_4706909
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4706909___redundant_placeholder05
1while_while_cond_4706909___redundant_placeholder15
1while_while_cond_4706909___redundant_placeholder25
1while_while_cond_4706909___redundant_placeholder3
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
while_cond_4703816
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_4703816___redundant_placeholder05
1while_while_cond_4703816___redundant_placeholder15
1while_while_cond_4703816___redundant_placeholder25
1while_while_cond_4703816___redundant_placeholder3
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
§
Í
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4706205

inputsX
Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource:4@U
Gforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource:4@V
Hbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp¢@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp¢backward_simple_rnn_9/while¢>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp¢?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp¢forward_simple_rnn_9/whileP
forward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"forward_simple_rnn_9/strided_sliceStridedSlice#forward_simple_rnn_9/Shape:output:01forward_simple_rnn_9/strided_slice/stack:output:03forward_simple_rnn_9/strided_slice/stack_1:output:03forward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@²
!forward_simple_rnn_9/zeros/packedPack+forward_simple_rnn_9/strided_slice:output:0,forward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
forward_simple_rnn_9/zerosFill*forward_simple_rnn_9/zeros/packed:output:0)forward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
#forward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_9/transpose	Transposeinputs,forward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4n
forward_simple_rnn_9/Shape_1Shape"forward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$forward_simple_rnn_9/strided_slice_1StridedSlice%forward_simple_rnn_9/Shape_1:output:03forward_simple_rnn_9/strided_slice_1/stack:output:05forward_simple_rnn_9/strided_slice_1/stack_1:output:05forward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"forward_simple_rnn_9/TensorArrayV2TensorListReserve9forward_simple_rnn_9/TensorArrayV2/element_shape:output:0-forward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
<forward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_9/transpose:y:0Sforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*forward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$forward_simple_rnn_9/strided_slice_2StridedSlice"forward_simple_rnn_9/transpose:y:03forward_simple_rnn_9/strided_slice_2/stack:output:05forward_simple_rnn_9/strided_slice_2/stack_1:output:05forward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÄ
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0à
.forward_simple_rnn_9/simple_rnn_cell_28/MatMulMatMul-forward_simple_rnn_9/strided_slice_2:output:0Eforward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
/forward_simple_rnn_9/simple_rnn_cell_28/BiasAddBiasAdd8forward_simple_rnn_9/simple_rnn_cell_28/MatMul:product:0Fforward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ú
0forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1MatMul#forward_simple_rnn_9/zeros:output:0Gforward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ü
+forward_simple_rnn_9/simple_rnn_cell_28/addAddV28forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd:output:0:forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,forward_simple_rnn_9/simple_rnn_cell_28/TanhTanh/forward_simple_rnn_9/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2forward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   s
1forward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$forward_simple_rnn_9/TensorArrayV2_1TensorListReserve;forward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0:forward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
forward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'forward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
forward_simple_rnn_9/whileWhile0forward_simple_rnn_9/while/loop_counter:output:06forward_simple_rnn_9/while/maximum_iterations:output:0"forward_simple_rnn_9/time:output:0-forward_simple_rnn_9/TensorArrayV2_1:handle:0#forward_simple_rnn_9/zeros:output:0-forward_simple_rnn_9/strided_slice_1:output:0Lforward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_9_simple_rnn_cell_28_matmul_readvariableop_resourceGforward_simple_rnn_9_simple_rnn_cell_28_biasadd_readvariableop_resourceHforward_simple_rnn_9_simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
'forward_simple_rnn_9_while_body_4706028*3
cond+R)
'forward_simple_rnn_9_while_cond_4706027*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Eforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
7forward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_9/while:output:3Nforward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements}
*forward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,forward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$forward_simple_rnn_9/strided_slice_3StridedSlice@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_9/strided_slice_3/stack:output:05forward_simple_rnn_9/strided_slice_3/stack_1:output:05forward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskz
%forward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 forward_simple_rnn_9/transpose_1	Transpose@forward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
backward_simple_rnn_9/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#backward_simple_rnn_9/strided_sliceStridedSlice$backward_simple_rnn_9/Shape:output:02backward_simple_rnn_9/strided_slice/stack:output:04backward_simple_rnn_9/strided_slice/stack_1:output:04backward_simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"backward_simple_rnn_9/zeros/packedPack,backward_simple_rnn_9/strided_slice:output:0-backward_simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
backward_simple_rnn_9/zerosFill+backward_simple_rnn_9/zeros/packed:output:0*backward_simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$backward_simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_simple_rnn_9/transpose	Transposeinputs-backward_simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
backward_simple_rnn_9/Shape_1Shape#backward_simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%backward_simple_rnn_9/strided_slice_1StridedSlice&backward_simple_rnn_9/Shape_1:output:04backward_simple_rnn_9/strided_slice_1/stack:output:06backward_simple_rnn_9/strided_slice_1/stack_1:output:06backward_simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#backward_simple_rnn_9/TensorArrayV2TensorListReserve:backward_simple_rnn_9/TensorArrayV2/element_shape:output:0.backward_simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_simple_rnn_9/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¶
backward_simple_rnn_9/ReverseV2	ReverseV2#backward_simple_rnn_9/transpose:y:0-backward_simple_rnn_9/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Kbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   §
=backward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_9/ReverseV2:output:0Tbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+backward_simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%backward_simple_rnn_9/strided_slice_2StridedSlice#backward_simple_rnn_9/transpose:y:04backward_simple_rnn_9/strided_slice_2/stack:output:06backward_simple_rnn_9/strided_slice_2/stack_1:output:06backward_simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/backward_simple_rnn_9/simple_rnn_cell_29/MatMulMatMul.backward_simple_rnn_9/strided_slice_2:output:0Fbackward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0backward_simple_rnn_9/simple_rnn_cell_29/BiasAddBiasAdd9backward_simple_rnn_9/simple_rnn_cell_29/MatMul:product:0Gbackward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1MatMul$backward_simple_rnn_9/zeros:output:0Hbackward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,backward_simple_rnn_9/simple_rnn_cell_29/addAddV29backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd:output:0;backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-backward_simple_rnn_9/simple_rnn_cell_29/TanhTanh0backward_simple_rnn_9/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3backward_simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2backward_simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%backward_simple_rnn_9/TensorArrayV2_1TensorListReserve<backward_simple_rnn_9/TensorArrayV2_1/element_shape:output:0;backward_simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
backward_simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(backward_simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
backward_simple_rnn_9/whileWhile1backward_simple_rnn_9/while/loop_counter:output:07backward_simple_rnn_9/while/maximum_iterations:output:0#backward_simple_rnn_9/time:output:0.backward_simple_rnn_9/TensorArrayV2_1:handle:0$backward_simple_rnn_9/zeros:output:0.backward_simple_rnn_9/strided_slice_1:output:0Mbackward_simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_9_simple_rnn_cell_29_matmul_readvariableop_resourceHbackward_simple_rnn_9_simple_rnn_cell_29_biasadd_readvariableop_resourceIbackward_simple_rnn_9_simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
(backward_simple_rnn_9_while_body_4706136*4
cond,R*
(backward_simple_rnn_9_while_cond_4706135*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8backward_simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_9/while:output:3Obackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+backward_simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-backward_simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%backward_simple_rnn_9/strided_slice_3StridedSliceAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_9/strided_slice_3/stack:output:06backward_simple_rnn_9/strided_slice_3/stack_1:output:06backward_simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&backward_simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!backward_simple_rnn_9/transpose_1	TransposeAbackward_simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ã
concatConcatV2-forward_simple_rnn_9/strided_slice_3:output:0.backward_simple_rnn_9/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp@^backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?^backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOpA^backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp^backward_simple_rnn_9/while?^forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>^forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp@^forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp^forward_simple_rnn_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp?backward_simple_rnn_9/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp>backward_simple_rnn_9/simple_rnn_cell_29/MatMul/ReadVariableOp2
@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp@backward_simple_rnn_9/simple_rnn_cell_29/MatMul_1/ReadVariableOp2:
backward_simple_rnn_9/whilebackward_simple_rnn_9/while2
>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp>forward_simple_rnn_9/simple_rnn_cell_28/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp=forward_simple_rnn_9/simple_rnn_cell_28/MatMul/ReadVariableOp2
?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp?forward_simple_rnn_9/simple_rnn_cell_28/MatMul_1/ReadVariableOp28
forward_simple_rnn_9/whileforward_simple_rnn_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
¥

÷
E__inference_dense_18_layer_call_and_return_conditional_losses_4706225

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

ì
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4707246

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
ÑR
è
9bidirectional_18_backward_simple_rnn_9_while_body_4704954j
fbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_loop_counterp
lbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_maximum_iterations<
8bidirectional_18_backward_simple_rnn_9_while_placeholder>
:bidirectional_18_backward_simple_rnn_9_while_placeholder_1>
:bidirectional_18_backward_simple_rnn_9_while_placeholder_2i
ebidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_strided_slice_1_0¦
¡bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0r
`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@o
abidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@t
bbidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@9
5bidirectional_18_backward_simple_rnn_9_while_identity;
7bidirectional_18_backward_simple_rnn_9_while_identity_1;
7bidirectional_18_backward_simple_rnn_9_while_identity_2;
7bidirectional_18_backward_simple_rnn_9_while_identity_3;
7bidirectional_18_backward_simple_rnn_9_while_identity_4g
cbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_strided_slice_1¤
bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorp
^bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource:4@m
_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource:@r
`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢Vbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Ubidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp¢Wbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp¯
^bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ê
Pbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¡bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_08bidirectional_18_backward_simple_rnn_9_while_placeholdergbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ö
Ubidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0º
Fbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMulMatMulWbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0]bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
Vbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpabidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¶
Gbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAddBiasAddPbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul:product:0^bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
Wbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpbbidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¡
Hbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1MatMul:bidirectional_18_backward_simple_rnn_9_while_placeholder_2_bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Cbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/addAddV2Pbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd:output:0Rbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
Dbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/TanhTanhGbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Wbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Qbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem:bidirectional_18_backward_simple_rnn_9_while_placeholder_1`bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0Hbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒt
2bidirectional_18/backward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ñ
0bidirectional_18/backward_simple_rnn_9/while/addAddV28bidirectional_18_backward_simple_rnn_9_while_placeholder;bidirectional_18/backward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: v
4bidirectional_18/backward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
2bidirectional_18/backward_simple_rnn_9/while/add_1AddV2fbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_loop_counter=bidirectional_18/backward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: Î
5bidirectional_18/backward_simple_rnn_9/while/IdentityIdentity6bidirectional_18/backward_simple_rnn_9/while/add_1:z:02^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
7bidirectional_18/backward_simple_rnn_9/while/Identity_1Identitylbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_maximum_iterations2^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Î
7bidirectional_18/backward_simple_rnn_9/while/Identity_2Identity4bidirectional_18/backward_simple_rnn_9/while/add:z:02^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: û
7bidirectional_18/backward_simple_rnn_9/while/Identity_3Identityabidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ó
7bidirectional_18/backward_simple_rnn_9/while/Identity_4IdentityHbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:02^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@þ
1bidirectional_18/backward_simple_rnn_9/while/NoOpNoOpW^bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpV^bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpX^bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ì
cbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_strided_slice_1ebidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_strided_slice_1_0"w
5bidirectional_18_backward_simple_rnn_9_while_identity>bidirectional_18/backward_simple_rnn_9/while/Identity:output:0"{
7bidirectional_18_backward_simple_rnn_9_while_identity_1@bidirectional_18/backward_simple_rnn_9/while/Identity_1:output:0"{
7bidirectional_18_backward_simple_rnn_9_while_identity_2@bidirectional_18/backward_simple_rnn_9/while/Identity_2:output:0"{
7bidirectional_18_backward_simple_rnn_9_while_identity_3@bidirectional_18/backward_simple_rnn_9/while/Identity_3:output:0"{
7bidirectional_18_backward_simple_rnn_9_while_identity_4@bidirectional_18/backward_simple_rnn_9/while/Identity_4:output:0"Ä
_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resourceabidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"Æ
`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resourcebbidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0"Â
^bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0"Æ
bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor¡bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2°
Vbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpVbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2®
Ubidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpUbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp2²
Wbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpWbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
(backward_simple_rnn_9_while_cond_4705475H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2J
Fbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705475___redundant_placeholder0a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705475___redundant_placeholder1a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705475___redundant_placeholder2a
]backward_simple_rnn_9_while_backward_simple_rnn_9_while_cond_4705475___redundant_placeholder3(
$backward_simple_rnn_9_while_identity
º
 backward_simple_rnn_9/while/LessLess'backward_simple_rnn_9_while_placeholderFbackward_simple_rnn_9_while_less_backward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_9/while/IdentityIdentity$backward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0*(
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
while_body_4703262
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_29_4703284_0:4@0
"while_simple_rnn_cell_29_4703286_0:@4
"while_simple_rnn_cell_29_4703288_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_29_4703284:4@.
 while_simple_rnn_cell_29_4703286:@2
 while_simple_rnn_cell_29_4703288:@@¢0while/simple_rnn_cell_29/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_29/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_29_4703284_0"while_simple_rnn_cell_29_4703286_0"while_simple_rnn_cell_29_4703288_0*
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4703248r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_29/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_29/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_29_4703284"while_simple_rnn_cell_29_4703284_0"F
 while_simple_rnn_cell_29_4703286"while_simple_rnn_cell_29_4703286_0"F
 while_simple_rnn_cell_29_4703288"while_simple_rnn_cell_29_4703288_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_29/StatefulPartitionedCall0while/simple_rnn_cell_29/StatefulPartitionedCall: 
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
Ê

*__inference_dense_18_layer_call_fn_4706214

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
E__inference_dense_18_layer_call_and_return_conditional_losses_4704313o
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
þ6
¬
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703326

inputs,
simple_rnn_cell_29_4703249:4@(
simple_rnn_cell_29_4703251:@,
simple_rnn_cell_29_4703253:@@
identity¢*simple_rnn_cell_29/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_29_4703249simple_rnn_cell_29_4703251simple_rnn_cell_29_4703253*
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4703248n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_29_4703249simple_rnn_cell_29_4703251simple_rnn_cell_29_4703253*
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
while_body_4703262*
condR
while_cond_4703261*8
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
NoOpNoOp+^simple_rnn_cell_29/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_29/StatefulPartitionedCall*simple_rnn_cell_29/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
½"
ß
while_body_4702964
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_28_4702986_0:4@0
"while_simple_rnn_cell_28_4702988_0:@4
"while_simple_rnn_cell_28_4702990_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_28_4702986:4@.
 while_simple_rnn_cell_28_4702988:@2
 while_simple_rnn_cell_28_4702990:@@¢0while/simple_rnn_cell_28/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_28/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_28_4702986_0"while_simple_rnn_cell_28_4702988_0"while_simple_rnn_cell_28_4702990_0*
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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4702950r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_28/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_28/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_28_4702986"while_simple_rnn_cell_28_4702986_0"F
 while_simple_rnn_cell_28_4702988"while_simple_rnn_cell_28_4702988_0"F
 while_simple_rnn_cell_28_4702990"while_simple_rnn_cell_28_4702990_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_28/StatefulPartitionedCall0while/simple_rnn_cell_28/StatefulPartitionedCall: 
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

	
9bidirectional_18_backward_simple_rnn_9_while_cond_4705180j
fbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_loop_counterp
lbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_maximum_iterations<
8bidirectional_18_backward_simple_rnn_9_while_placeholder>
:bidirectional_18_backward_simple_rnn_9_while_placeholder_1>
:bidirectional_18_backward_simple_rnn_9_while_placeholder_2l
hbidirectional_18_backward_simple_rnn_9_while_less_bidirectional_18_backward_simple_rnn_9_strided_slice_1
bidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_cond_4705180___redundant_placeholder0
bidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_cond_4705180___redundant_placeholder1
bidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_cond_4705180___redundant_placeholder2
bidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_cond_4705180___redundant_placeholder39
5bidirectional_18_backward_simple_rnn_9_while_identity
þ
1bidirectional_18/backward_simple_rnn_9/while/LessLess8bidirectional_18_backward_simple_rnn_9_while_placeholderhbidirectional_18_backward_simple_rnn_9_while_less_bidirectional_18_backward_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: 
5bidirectional_18/backward_simple_rnn_9/while/IdentityIdentity5bidirectional_18/backward_simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "w
5bidirectional_18_backward_simple_rnn_9_while_identity>bidirectional_18/backward_simple_rnn_9/while/Identity:output:0*(
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
(backward_simple_rnn_9_while_body_4704519H
Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counterN
Jbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations+
'backward_simple_rnn_9_while_placeholder-
)backward_simple_rnn_9_while_placeholder_1-
)backward_simple_rnn_9_while_placeholder_2G
Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0
backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@^
Pbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_9_while_identity*
&backward_simple_rnn_9_while_identity_1*
&backward_simple_rnn_9_while_identity_2*
&backward_simple_rnn_9_while_identity_3*
&backward_simple_rnn_9_while_identity_4E
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource:4@\
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp¢Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
Mbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_9_while_placeholderVbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5backward_simple_rnn_9/while/simple_rnn_cell_29/MatMulMatMulFbackward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAddBiasAdd?backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul:product:0Mbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1MatMul)backward_simple_rnn_9_while_placeholder_2Nbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2backward_simple_rnn_9/while/simple_rnn_cell_29/addAddV2?backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd:output:0Abackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3backward_simple_rnn_9/while/simple_rnn_cell_29/TanhTanh6backward_simple_rnn_9/while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_9_while_placeholder_1Obackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:07backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!backward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_simple_rnn_9/while/addAddV2'backward_simple_rnn_9_while_placeholder*backward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!backward_simple_rnn_9/while/add_1AddV2Dbackward_simple_rnn_9_while_backward_simple_rnn_9_while_loop_counter,backward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
$backward_simple_rnn_9/while/IdentityIdentity%backward_simple_rnn_9/while/add_1:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Â
&backward_simple_rnn_9/while/Identity_1IdentityJbackward_simple_rnn_9_while_backward_simple_rnn_9_while_maximum_iterations!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
&backward_simple_rnn_9/while/Identity_2Identity#backward_simple_rnn_9/while/add:z:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: È
&backward_simple_rnn_9/while/Identity_3IdentityPbackward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: À
&backward_simple_rnn_9/while/Identity_4Identity7backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0!^backward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 backward_simple_rnn_9/while/NoOpNoOpF^backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpE^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpG^backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Abackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1Cbackward_simple_rnn_9_while_backward_simple_rnn_9_strided_slice_1_0"U
$backward_simple_rnn_9_while_identity-backward_simple_rnn_9/while/Identity:output:0"Y
&backward_simple_rnn_9_while_identity_1/backward_simple_rnn_9/while/Identity_1:output:0"Y
&backward_simple_rnn_9_while_identity_2/backward_simple_rnn_9/while/Identity_2:output:0"Y
&backward_simple_rnn_9_while_identity_3/backward_simple_rnn_9/while/Identity_3:output:0"Y
&backward_simple_rnn_9_while_identity_4/backward_simple_rnn_9/while/Identity_4:output:0"¢
Nbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resourcePbackward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"¤
Obackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resourceQbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0" 
Mbackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resourceObackward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0"
}backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ebackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpEbackward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2
Dbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpDbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp2
Fbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpFbackward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
while_body_4707022
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_29_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_29/MatMul/ReadVariableOp¢0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_29/BiasAddBiasAdd)while/simple_rnn_cell_29/MatMul:product:07while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_29/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_29/addAddV2)while/simple_rnn_cell_29/BiasAdd:output:0+while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_29/TanhTanh while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_29/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_29/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_29/MatMul/ReadVariableOp1^while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_29_matmul_readvariableop_resource9while_simple_rnn_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_29/MatMul/ReadVariableOp.while/simple_rnn_cell_29/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
while_body_4703666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_29_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_29/MatMul/ReadVariableOp¢0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_29/BiasAddBiasAdd)while/simple_rnn_cell_29/MatMul:product:07while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_29/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_29/addAddV2)while/simple_rnn_cell_29/BiasAdd:output:0+while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_29/TanhTanh while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_29/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_29/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_29/MatMul/ReadVariableOp1^while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_29_biasadd_readvariableop_resource:while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_29_matmul_1_readvariableop_resource;while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_29_matmul_readvariableop_resource9while_simple_rnn_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_29/MatMul/ReadVariableOp.while/simple_rnn_cell_29/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp0while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
­
Ã
7__inference_backward_simple_rnn_9_layer_call_fn_4706720
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703326o
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

ì
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4707325

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
´
ü
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704648

inputs*
bidirectional_18_4704629:4@&
bidirectional_18_4704631:@*
bidirectional_18_4704633:@@*
bidirectional_18_4704635:4@&
bidirectional_18_4704637:@*
bidirectional_18_4704639:@@#
dense_18_4704642:	
dense_18_4704644:
identity¢(bidirectional_18/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall
(bidirectional_18/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_18_4704629bidirectional_18_4704631bidirectional_18_4704633bidirectional_18_4704635bidirectional_18_4704637bidirectional_18_4704639*
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4704588¡
 dense_18/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_18/StatefulPartitionedCall:output:0dense_18_4704642dense_18_4704644*
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
E__inference_dense_18_layer_call_and_return_conditional_losses_4704313x
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_18/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_18/StatefulPartitionedCall(bidirectional_18/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ÿ@
Ë
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4707201

inputsC
1simple_rnn_cell_29_matmul_readvariableop_resource:4@@
2simple_rnn_cell_29_biasadd_readvariableop_resource:@E
3simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_29/BiasAdd/ReadVariableOp¢(simple_rnn_cell_29/MatMul/ReadVariableOp¢*simple_rnn_cell_29/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_29/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_29/BiasAddBiasAdd#simple_rnn_cell_29/MatMul:product:01simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_29/MatMul_1MatMulzeros:output:02simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_29/addAddV2#simple_rnn_cell_29/BiasAdd:output:0%simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_29/TanhTanhsimple_rnn_cell_29/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_29_matmul_readvariableop_resource2simple_rnn_cell_29_biasadd_readvariableop_resource3simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
while_body_4707134*
condR
while_cond_4707133*8
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
NoOpNoOp*^simple_rnn_cell_29/BiasAdd/ReadVariableOp)^simple_rnn_cell_29/MatMul/ReadVariableOp+^simple_rnn_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_29/BiasAdd/ReadVariableOp)simple_rnn_cell_29/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_29/MatMul/ReadVariableOp(simple_rnn_cell_29/MatMul/ReadVariableOp2X
*simple_rnn_cell_29/MatMul_1/ReadVariableOp*simple_rnn_cell_29/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
äA
É
'forward_simple_rnn_9_while_body_4705588F
Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counterL
Hforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations*
&forward_simple_rnn_9_while_placeholder,
(forward_simple_rnn_9_while_placeholder_1,
(forward_simple_rnn_9_while_placeholder_2E
Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0
}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@]
Oforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_9_while_identity)
%forward_simple_rnn_9_while_identity_1)
%forward_simple_rnn_9_while_identity_2)
%forward_simple_rnn_9_while_identity_3)
%forward_simple_rnn_9_while_identity_4C
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource:4@[
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp¢Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
Lforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
>forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_9_while_placeholderUforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ò
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
4forward_simple_rnn_9/while/simple_rnn_cell_28/MatMulMatMulEforward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
5forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAddBiasAdd>forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul:product:0Lforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ë
6forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1MatMul(forward_simple_rnn_9_while_placeholder_2Mforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
1forward_simple_rnn_9/while/simple_rnn_cell_28/addAddV2>forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd:output:0@forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
2forward_simple_rnn_9/while/simple_rnn_cell_28/TanhTanh5forward_simple_rnn_9/while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Eforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
?forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_9_while_placeholder_1Nforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:06forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒb
 forward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_9/while/addAddV2&forward_simple_rnn_9_while_placeholder)forward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 forward_simple_rnn_9/while/add_1AddV2Bforward_simple_rnn_9_while_forward_simple_rnn_9_while_loop_counter+forward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
#forward_simple_rnn_9/while/IdentityIdentity$forward_simple_rnn_9/while/add_1:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ¾
%forward_simple_rnn_9/while/Identity_1IdentityHforward_simple_rnn_9_while_forward_simple_rnn_9_while_maximum_iterations ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
%forward_simple_rnn_9/while/Identity_2Identity"forward_simple_rnn_9/while/add:z:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Å
%forward_simple_rnn_9/while/Identity_3IdentityOforward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ½
%forward_simple_rnn_9/while/Identity_4Identity6forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0 ^forward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
forward_simple_rnn_9/while/NoOpNoOpE^forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpD^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpF^forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
?forward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1Aforward_simple_rnn_9_while_forward_simple_rnn_9_strided_slice_1_0"S
#forward_simple_rnn_9_while_identity,forward_simple_rnn_9/while/Identity:output:0"W
%forward_simple_rnn_9_while_identity_1.forward_simple_rnn_9/while/Identity_1:output:0"W
%forward_simple_rnn_9_while_identity_2.forward_simple_rnn_9/while/Identity_2:output:0"W
%forward_simple_rnn_9_while_identity_3.forward_simple_rnn_9/while/Identity_3:output:0"W
%forward_simple_rnn_9_while_identity_4.forward_simple_rnn_9/while/Identity_4:output:0" 
Mforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resourceOforward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"¢
Nforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resourcePforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"
Lforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resourceNforward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0"ü
{forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Dforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpDforward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2
Cforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpCforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp2
Eforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpEforward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4703614

inputsC
1simple_rnn_cell_28_matmul_readvariableop_resource:4@@
2simple_rnn_cell_28_biasadd_readvariableop_resource:@E
3simple_rnn_cell_28_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_28/BiasAdd/ReadVariableOp¢(simple_rnn_cell_28/MatMul/ReadVariableOp¢*simple_rnn_cell_28/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_28/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_28/BiasAddBiasAdd#simple_rnn_cell_28/MatMul:product:01simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_28/MatMul_1MatMulzeros:output:02simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_28/addAddV2#simple_rnn_cell_28/BiasAdd:output:0%simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_28/TanhTanhsimple_rnn_cell_28/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_28_matmul_readvariableop_resource2simple_rnn_cell_28_biasadd_readvariableop_resource3simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
while_body_4703547*
condR
while_cond_4703546*8
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
NoOpNoOp*^simple_rnn_cell_28/BiasAdd/ReadVariableOp)^simple_rnn_cell_28/MatMul/ReadVariableOp+^simple_rnn_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_28/BiasAdd/ReadVariableOp)simple_rnn_cell_28/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_28/MatMul/ReadVariableOp(simple_rnn_cell_28/MatMul/ReadVariableOp2X
*simple_rnn_cell_28/MatMul_1/ReadVariableOp*simple_rnn_cell_28/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
æ
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4703744

inputs.
forward_simple_rnn_9_4703615:4@*
forward_simple_rnn_9_4703617:@.
forward_simple_rnn_9_4703619:@@/
backward_simple_rnn_9_4703734:4@+
backward_simple_rnn_9_4703736:@/
backward_simple_rnn_9_4703738:@@
identity¢-backward_simple_rnn_9/StatefulPartitionedCall¢,forward_simple_rnn_9/StatefulPartitionedCallÆ
,forward_simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_9_4703615forward_simple_rnn_9_4703617forward_simple_rnn_9_4703619*
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4703614Ë
-backward_simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_9_4703734backward_simple_rnn_9_4703736backward_simple_rnn_9_4703738*
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4703733M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ó
concatConcatV25forward_simple_rnn_9/StatefulPartitionedCall:output:06backward_simple_rnn_9/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp.^backward_simple_rnn_9/StatefulPartitionedCall-^forward_simple_rnn_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_simple_rnn_9/StatefulPartitionedCall-backward_simple_rnn_9/StatefulPartitionedCall2\
,forward_simple_rnn_9/StatefulPartitionedCall,forward_simple_rnn_9/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ@
Ë
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4707089

inputsC
1simple_rnn_cell_29_matmul_readvariableop_resource:4@@
2simple_rnn_cell_29_biasadd_readvariableop_resource:@E
3simple_rnn_cell_29_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_29/BiasAdd/ReadVariableOp¢(simple_rnn_cell_29/MatMul/ReadVariableOp¢*simple_rnn_cell_29/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_29_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_29/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_29/BiasAddBiasAdd#simple_rnn_cell_29/MatMul:product:01simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_29_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_29/MatMul_1MatMulzeros:output:02simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_29/addAddV2#simple_rnn_cell_29/BiasAdd:output:0%simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_29/TanhTanhsimple_rnn_cell_29/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_29_matmul_readvariableop_resource2simple_rnn_cell_29_biasadd_readvariableop_resource3simple_rnn_cell_29_matmul_1_readvariableop_resource*
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
while_body_4707022*
condR
while_cond_4707021*8
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
NoOpNoOp*^simple_rnn_cell_29/BiasAdd/ReadVariableOp)^simple_rnn_cell_29/MatMul/ReadVariableOp+^simple_rnn_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_29/BiasAdd/ReadVariableOp)simple_rnn_cell_29/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_29/MatMul/ReadVariableOp(simple_rnn_cell_29/MatMul/ReadVariableOp2X
*simple_rnn_cell_29/MatMul_1/ReadVariableOp*simple_rnn_cell_29/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÙQ
Ê
8bidirectional_18_forward_simple_rnn_9_while_body_4705073h
dbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_loop_countern
jbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_maximum_iterations;
7bidirectional_18_forward_simple_rnn_9_while_placeholder=
9bidirectional_18_forward_simple_rnn_9_while_placeholder_1=
9bidirectional_18_forward_simple_rnn_9_while_placeholder_2g
cbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_strided_slice_1_0¤
bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0q
_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@n
`bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@s
abidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@8
4bidirectional_18_forward_simple_rnn_9_while_identity:
6bidirectional_18_forward_simple_rnn_9_while_identity_1:
6bidirectional_18_forward_simple_rnn_9_while_identity_2:
6bidirectional_18_forward_simple_rnn_9_while_identity_3:
6bidirectional_18_forward_simple_rnn_9_while_identity_4e
abidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_strided_slice_1¢
bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensoro
]bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource:4@l
^bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource:@q
_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢Ubidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢Tbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp¢Vbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp®
]bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   å
Obidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_07bidirectional_18_forward_simple_rnn_9_while_placeholderfbidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ô
Tbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0·
Ebidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMulMatMulVbidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0\bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ò
Ubidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp`bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0³
Fbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAddBiasAddObidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul:product:0]bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ø
Vbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOpabidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0
Gbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1MatMul9bidirectional_18_forward_simple_rnn_9_while_placeholder_2^bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
Bbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/addAddV2Obidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd:output:0Qbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
Cbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/TanhTanhFbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Vbidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Pbidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem9bidirectional_18_forward_simple_rnn_9_while_placeholder_1_bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0Gbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒs
1bidirectional_18/forward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Î
/bidirectional_18/forward_simple_rnn_9/while/addAddV27bidirectional_18_forward_simple_rnn_9_while_placeholder:bidirectional_18/forward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: u
3bidirectional_18/forward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ÿ
1bidirectional_18/forward_simple_rnn_9/while/add_1AddV2dbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_loop_counter<bidirectional_18/forward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: Ë
4bidirectional_18/forward_simple_rnn_9/while/IdentityIdentity5bidirectional_18/forward_simple_rnn_9/while/add_1:z:01^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
6bidirectional_18/forward_simple_rnn_9/while/Identity_1Identityjbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_while_maximum_iterations1^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Ë
6bidirectional_18/forward_simple_rnn_9/while/Identity_2Identity3bidirectional_18/forward_simple_rnn_9/while/add:z:01^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ø
6bidirectional_18/forward_simple_rnn_9/while/Identity_3Identity`bidirectional_18/forward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ð
6bidirectional_18/forward_simple_rnn_9/while/Identity_4IdentityGbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/Tanh:y:01^bidirectional_18/forward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
0bidirectional_18/forward_simple_rnn_9/while/NoOpNoOpV^bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpU^bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpW^bidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "È
abidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_strided_slice_1cbidirectional_18_forward_simple_rnn_9_while_bidirectional_18_forward_simple_rnn_9_strided_slice_1_0"u
4bidirectional_18_forward_simple_rnn_9_while_identity=bidirectional_18/forward_simple_rnn_9/while/Identity:output:0"y
6bidirectional_18_forward_simple_rnn_9_while_identity_1?bidirectional_18/forward_simple_rnn_9/while/Identity_1:output:0"y
6bidirectional_18_forward_simple_rnn_9_while_identity_2?bidirectional_18/forward_simple_rnn_9/while/Identity_2:output:0"y
6bidirectional_18_forward_simple_rnn_9_while_identity_3?bidirectional_18/forward_simple_rnn_9/while/Identity_3:output:0"y
6bidirectional_18_forward_simple_rnn_9_while_identity_4?bidirectional_18/forward_simple_rnn_9/while/Identity_4:output:0"Â
^bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource`bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"Ä
_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resourceabidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"À
]bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_bidirectional_18_forward_simple_rnn_9_while_simple_rnn_cell_28_matmul_readvariableop_resource_0"Â
bidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorbidirectional_18_forward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_forward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2®
Ubidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpUbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2¬
Tbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOpTbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul/ReadVariableOp2°
Vbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOpVbidirectional_18/forward_simple_rnn_9/while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
ÑR
è
9bidirectional_18_backward_simple_rnn_9_while_body_4705181j
fbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_loop_counterp
lbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_maximum_iterations<
8bidirectional_18_backward_simple_rnn_9_while_placeholder>
:bidirectional_18_backward_simple_rnn_9_while_placeholder_1>
:bidirectional_18_backward_simple_rnn_9_while_placeholder_2i
ebidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_strided_slice_1_0¦
¡bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0r
`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0:4@o
abidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0:@t
bbidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0:@@9
5bidirectional_18_backward_simple_rnn_9_while_identity;
7bidirectional_18_backward_simple_rnn_9_while_identity_1;
7bidirectional_18_backward_simple_rnn_9_while_identity_2;
7bidirectional_18_backward_simple_rnn_9_while_identity_3;
7bidirectional_18_backward_simple_rnn_9_while_identity_4g
cbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_strided_slice_1¤
bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorp
^bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource:4@m
_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource:@r
`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource:@@¢Vbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp¢Ubidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp¢Wbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp¯
^bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ê
Pbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¡bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_08bidirectional_18_backward_simple_rnn_9_while_placeholdergbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ö
Ubidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpReadVariableOp`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0º
Fbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMulMatMulWbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0]bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
Vbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpReadVariableOpabidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¶
Gbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAddBiasAddPbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul:product:0^bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
Wbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpReadVariableOpbbidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¡
Hbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1MatMul:bidirectional_18_backward_simple_rnn_9_while_placeholder_2_bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Cbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/addAddV2Pbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd:output:0Rbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
Dbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/TanhTanhGbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Wbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Qbidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem:bidirectional_18_backward_simple_rnn_9_while_placeholder_1`bidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0Hbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒt
2bidirectional_18/backward_simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ñ
0bidirectional_18/backward_simple_rnn_9/while/addAddV28bidirectional_18_backward_simple_rnn_9_while_placeholder;bidirectional_18/backward_simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: v
4bidirectional_18/backward_simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
2bidirectional_18/backward_simple_rnn_9/while/add_1AddV2fbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_loop_counter=bidirectional_18/backward_simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: Î
5bidirectional_18/backward_simple_rnn_9/while/IdentityIdentity6bidirectional_18/backward_simple_rnn_9/while/add_1:z:02^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: 
7bidirectional_18/backward_simple_rnn_9/while/Identity_1Identitylbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_while_maximum_iterations2^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Î
7bidirectional_18/backward_simple_rnn_9/while/Identity_2Identity4bidirectional_18/backward_simple_rnn_9/while/add:z:02^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: û
7bidirectional_18/backward_simple_rnn_9/while/Identity_3Identityabidirectional_18/backward_simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ó
7bidirectional_18/backward_simple_rnn_9/while/Identity_4IdentityHbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/Tanh:y:02^bidirectional_18/backward_simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@þ
1bidirectional_18/backward_simple_rnn_9/while/NoOpNoOpW^bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpV^bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpX^bidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ì
cbidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_strided_slice_1ebidirectional_18_backward_simple_rnn_9_while_bidirectional_18_backward_simple_rnn_9_strided_slice_1_0"w
5bidirectional_18_backward_simple_rnn_9_while_identity>bidirectional_18/backward_simple_rnn_9/while/Identity:output:0"{
7bidirectional_18_backward_simple_rnn_9_while_identity_1@bidirectional_18/backward_simple_rnn_9/while/Identity_1:output:0"{
7bidirectional_18_backward_simple_rnn_9_while_identity_2@bidirectional_18/backward_simple_rnn_9/while/Identity_2:output:0"{
7bidirectional_18_backward_simple_rnn_9_while_identity_3@bidirectional_18/backward_simple_rnn_9/while/Identity_3:output:0"{
7bidirectional_18_backward_simple_rnn_9_while_identity_4@bidirectional_18/backward_simple_rnn_9/while/Identity_4:output:0"Ä
_bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resourceabidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_biasadd_readvariableop_resource_0"Æ
`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resourcebbidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_1_readvariableop_resource_0"Â
^bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource`bidirectional_18_backward_simple_rnn_9_while_simple_rnn_cell_29_matmul_readvariableop_resource_0"Æ
bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor¡bidirectional_18_backward_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_bidirectional_18_backward_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2°
Vbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOpVbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/BiasAdd/ReadVariableOp2®
Ubidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOpUbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul/ReadVariableOp2²
Wbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOpWbidirectional_18/backward_simple_rnn_9/while/simple_rnn_cell_29/MatMul_1/ReadVariableOp: 
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
while_body_4703949
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_28_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_28_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_28/MatMul/ReadVariableOp¢0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_28_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_28/BiasAddBiasAdd)while/simple_rnn_cell_28/MatMul:product:07while/simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_28/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_28/addAddV2)while/simple_rnn_cell_28/BiasAdd:output:0+while/simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_28/TanhTanh while/simple_rnn_cell_28/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_28/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_28/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_28/MatMul/ReadVariableOp1^while/simple_rnn_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_28_biasadd_readvariableop_resource:while_simple_rnn_cell_28_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_28_matmul_1_readvariableop_resource;while_simple_rnn_cell_28_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_28_matmul_readvariableop_resource9while_simple_rnn_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp/while/simple_rnn_cell_28/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_28/MatMul/ReadVariableOp.while/simple_rnn_cell_28/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp0while/simple_rnn_cell_28/MatMul_1/ReadVariableOp: 
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706709

inputsC
1simple_rnn_cell_28_matmul_readvariableop_resource:4@@
2simple_rnn_cell_28_biasadd_readvariableop_resource:@E
3simple_rnn_cell_28_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_28/BiasAdd/ReadVariableOp¢(simple_rnn_cell_28/MatMul/ReadVariableOp¢*simple_rnn_cell_28/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_28/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_28_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_28/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_28/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_28/BiasAddBiasAdd#simple_rnn_cell_28/MatMul:product:01simple_rnn_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_28/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_28_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_28/MatMul_1MatMulzeros:output:02simple_rnn_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_28/addAddV2#simple_rnn_cell_28/BiasAdd:output:0%simple_rnn_cell_28/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_28/TanhTanhsimple_rnn_cell_28/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_28_matmul_readvariableop_resource2simple_rnn_cell_28_biasadd_readvariableop_resource3simple_rnn_cell_28_matmul_1_readvariableop_resource*
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
while_body_4706642*
condR
while_cond_4706641*8
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
NoOpNoOp*^simple_rnn_cell_28/BiasAdd/ReadVariableOp)^simple_rnn_cell_28/MatMul/ReadVariableOp+^simple_rnn_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_28/BiasAdd/ReadVariableOp)simple_rnn_cell_28/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_28/MatMul/ReadVariableOp(simple_rnn_cell_28/MatMul/ReadVariableOp2X
*simple_rnn_cell_28/MatMul_1/ReadVariableOp*simple_rnn_cell_28/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Í
serving_default¹
]
bidirectional_18_inputC
(serving_default_bidirectional_18_input:0ÿÿÿÿÿÿÿÿÿ4<
dense_180
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
/__inference_sequential_18_layer_call_fn_4704339
/__inference_sequential_18_layer_call_fn_4704782
/__inference_sequential_18_layer_call_fn_4704803
/__inference_sequential_18_layer_call_fn_4704688¿
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4705030
J__inference_sequential_18_layer_call_and_return_conditional_losses_4705257
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704710
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704732¿
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
"__inference__wrapped_model_4702902bidirectional_18_input"
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
2__inference_bidirectional_18_layer_call_fn_4705274
2__inference_bidirectional_18_layer_call_fn_4705291
2__inference_bidirectional_18_layer_call_fn_4705308
2__inference_bidirectional_18_layer_call_fn_4705325å
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705545
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705765
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705985
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4706205å
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
*__inference_dense_18_layer_call_fn_4706214¢
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
E__inference_dense_18_layer_call_and_return_conditional_losses_4706225¢
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
": 	2dense_18/kernel
:2dense_18/bias
Q:O4@2?bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel
[:Y@@2Ibidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel
K:I@2=bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias
R:P4@2@bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel
\:Z@@2Jbidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel
L:J@2>bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias
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
/__inference_sequential_18_layer_call_fn_4704339bidirectional_18_input"¿
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
/__inference_sequential_18_layer_call_fn_4704782inputs"¿
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
/__inference_sequential_18_layer_call_fn_4704803inputs"¿
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
/__inference_sequential_18_layer_call_fn_4704688bidirectional_18_input"¿
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4705030inputs"¿
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4705257inputs"¿
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704710bidirectional_18_input"¿
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704732bidirectional_18_input"¿
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
%__inference_signature_wrapper_4704761bidirectional_18_input"
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
2__inference_bidirectional_18_layer_call_fn_4705274inputs/0"å
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
2__inference_bidirectional_18_layer_call_fn_4705291inputs/0"å
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
2__inference_bidirectional_18_layer_call_fn_4705308inputs"å
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
2__inference_bidirectional_18_layer_call_fn_4705325inputs"å
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705545inputs/0"å
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705765inputs/0"å
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705985inputs"å
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4706205inputs"å
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706236
6__inference_forward_simple_rnn_9_layer_call_fn_4706247
6__inference_forward_simple_rnn_9_layer_call_fn_4706258
6__inference_forward_simple_rnn_9_layer_call_fn_4706269Ô
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706379
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706489
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706599
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706709Ô
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
7__inference_backward_simple_rnn_9_layer_call_fn_4706720
7__inference_backward_simple_rnn_9_layer_call_fn_4706731
7__inference_backward_simple_rnn_9_layer_call_fn_4706742
7__inference_backward_simple_rnn_9_layer_call_fn_4706753Ô
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4706865
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4706977
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4707089
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4707201Ô
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
*__inference_dense_18_layer_call_fn_4706214inputs"¢
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
E__inference_dense_18_layer_call_and_return_conditional_losses_4706225inputs"¢
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706236inputs/0"Ô
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706247inputs/0"Ô
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706258inputs"Ô
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706269inputs"Ô
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706379inputs/0"Ô
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706489inputs/0"Ô
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706599inputs"Ô
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706709inputs"Ô
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
4__inference_simple_rnn_cell_28_layer_call_fn_4707215
4__inference_simple_rnn_cell_28_layer_call_fn_4707229½
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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4707246
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4707263½
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
7__inference_backward_simple_rnn_9_layer_call_fn_4706720inputs/0"Ô
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
7__inference_backward_simple_rnn_9_layer_call_fn_4706731inputs/0"Ô
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
7__inference_backward_simple_rnn_9_layer_call_fn_4706742inputs"Ô
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
7__inference_backward_simple_rnn_9_layer_call_fn_4706753inputs"Ô
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4706865inputs/0"Ô
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4706977inputs/0"Ô
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4707089inputs"Ô
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4707201inputs"Ô
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
4__inference_simple_rnn_cell_29_layer_call_fn_4707277
4__inference_simple_rnn_cell_29_layer_call_fn_4707291½
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4707308
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4707325½
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
4__inference_simple_rnn_cell_28_layer_call_fn_4707215inputsstates/0"½
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
4__inference_simple_rnn_cell_28_layer_call_fn_4707229inputsstates/0"½
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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4707246inputsstates/0"½
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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4707263inputsstates/0"½
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
4__inference_simple_rnn_cell_29_layer_call_fn_4707277inputsstates/0"½
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
4__inference_simple_rnn_cell_29_layer_call_fn_4707291inputsstates/0"½
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4707308inputsstates/0"½
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4707325inputsstates/0"½
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
':%	2Adam/dense_18/kernel/m
 :2Adam/dense_18/bias/m
V:T4@2FAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/m
`:^@@2PAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/m
P:N@2DAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/m
W:U4@2GAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/m
a:_@@2QAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/m
Q:O@2EAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/m
':%	2Adam/dense_18/kernel/v
 :2Adam/dense_18/bias/v
V:T4@2FAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/kernel/v
`:^@@2PAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/recurrent_kernel/v
P:N@2DAdam/bidirectional_18/forward_simple_rnn_9/simple_rnn_cell_28/bias/v
W:U4@2GAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/kernel/v
a:_@@2QAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/recurrent_kernel/v
Q:O@2EAdam/bidirectional_18/backward_simple_rnn_9/simple_rnn_cell_29/bias/v«
"__inference__wrapped_model_4702902! C¢@
9¢6
41
bidirectional_18_inputÿÿÿÿÿÿÿÿÿ4
ª "3ª0
.
dense_18"
dense_18ÿÿÿÿÿÿÿÿÿÓ
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4706865}! O¢L
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4706977}! O¢L
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4707089! Q¢N
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
R__inference_backward_simple_rnn_9_layer_call_and_return_conditional_losses_4707201! Q¢N
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
7__inference_backward_simple_rnn_9_layer_call_fn_4706720p! O¢L
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
7__inference_backward_simple_rnn_9_layer_call_fn_4706731p! O¢L
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
7__inference_backward_simple_rnn_9_layer_call_fn_4706742r! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@­
7__inference_backward_simple_rnn_9_layer_call_fn_4706753r! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@à
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705545! \¢Y
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705765! \¢Y
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4705985u! C¢@
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
M__inference_bidirectional_18_layer_call_and_return_conditional_losses_4706205u! C¢@
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
2__inference_bidirectional_18_layer_call_fn_4705274! \¢Y
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
2__inference_bidirectional_18_layer_call_fn_4705291! \¢Y
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
2__inference_bidirectional_18_layer_call_fn_4705308h! C¢@
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
2__inference_bidirectional_18_layer_call_fn_4705325h! C¢@
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
E__inference_dense_18_layer_call_and_return_conditional_losses_4706225]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_dense_18_layer_call_fn_4706214P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÒ
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706379}O¢L
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706489}O¢L
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706599Q¢N
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
Q__inference_forward_simple_rnn_9_layer_call_and_return_conditional_losses_4706709Q¢N
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706236pO¢L
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706247pO¢L
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
6__inference_forward_simple_rnn_9_layer_call_fn_4706258rQ¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@¬
6__inference_forward_simple_rnn_9_layer_call_fn_4706269rQ¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@Ì
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704710~! K¢H
A¢>
41
bidirectional_18_inputÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
J__inference_sequential_18_layer_call_and_return_conditional_losses_4704732~! K¢H
A¢>
41
bidirectional_18_inputÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
J__inference_sequential_18_layer_call_and_return_conditional_losses_4705030n! ;¢8
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_4705257n! ;¢8
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
/__inference_sequential_18_layer_call_fn_4704339q! K¢H
A¢>
41
bidirectional_18_inputÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
/__inference_sequential_18_layer_call_fn_4704688q! K¢H
A¢>
41
bidirectional_18_inputÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_18_layer_call_fn_4704782a! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_18_layer_call_fn_4704803a! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ
%__inference_signature_wrapper_4704761! ]¢Z
¢ 
SªP
N
bidirectional_18_input41
bidirectional_18_inputÿÿÿÿÿÿÿÿÿ4"3ª0
.
dense_18"
dense_18ÿÿÿÿÿÿÿÿÿ
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4707246·\¢Y
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
O__inference_simple_rnn_cell_28_layer_call_and_return_conditional_losses_4707263·\¢Y
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
4__inference_simple_rnn_cell_28_layer_call_fn_4707215©\¢Y
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
4__inference_simple_rnn_cell_28_layer_call_fn_4707229©\¢Y
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4707308·! \¢Y
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
O__inference_simple_rnn_cell_29_layer_call_and_return_conditional_losses_4707325·! \¢Y
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
4__inference_simple_rnn_cell_29_layer_call_fn_4707277©! \¢Y
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
4__inference_simple_rnn_cell_29_layer_call_fn_4707291©! \¢Y
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