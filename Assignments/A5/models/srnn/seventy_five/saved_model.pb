û1
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
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ðÈ.
ä
FAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*W
shared_nameHFAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/v
Ý
ZAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/v/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/v*
_output_shapes
:@*
dtype0

RAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*c
shared_nameTRAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/v
ù
fAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpRAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
ì
HAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*Y
shared_nameJHAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/v
å
\Adam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/v/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/v*
_output_shapes

:4@*
dtype0
â
EAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*V
shared_nameGEAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/v
Û
YAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/v/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/v*
_output_shapes
:@*
dtype0
þ
QAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*b
shared_nameSQAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/v
÷
eAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpQAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
ê
GAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*X
shared_nameIGAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/v
ã
[Adam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/v*
_output_shapes

:4@*
dtype0

Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:*
dtype0

Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_21/kernel/v

*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	*
dtype0
ä
FAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*W
shared_nameHFAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/m
Ý
ZAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/m/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/m*
_output_shapes
:@*
dtype0

RAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*c
shared_nameTRAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/m
ù
fAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpRAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
ì
HAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*Y
shared_nameJHAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/m
å
\Adam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/m/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/m*
_output_shapes

:4@*
dtype0
â
EAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*V
shared_nameGEAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/m
Û
YAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/m/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/m*
_output_shapes
:@*
dtype0
þ
QAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*b
shared_nameSQAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/m
÷
eAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpQAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
ê
GAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*X
shared_nameIGAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/m
ã
[Adam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/m*
_output_shapes

:4@*
dtype0

Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:*
dtype0

Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_21/kernel/m

*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
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
Ö
?bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias
Ï
Sbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/Read/ReadVariableOpReadVariableOp?bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias*
_output_shapes
:@*
dtype0
ò
Kbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*\
shared_nameMKbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel
ë
_bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/Read/ReadVariableOpReadVariableOpKbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel*
_output_shapes

:@@*
dtype0
Þ
Abidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*R
shared_nameCAbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel
×
Ubidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/Read/ReadVariableOpReadVariableOpAbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel*
_output_shapes

:4@*
dtype0
Ô
>bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*O
shared_name@>bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias
Í
Rbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/Read/ReadVariableOpReadVariableOp>bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias*
_output_shapes
:@*
dtype0
ð
Jbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*[
shared_nameLJbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel
é
^bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/Read/ReadVariableOpReadVariableOpJbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel*
_output_shapes

:@@*
dtype0
Ü
@bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*Q
shared_nameB@bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel
Õ
Tbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/Read/ReadVariableOpReadVariableOp@bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel*
_output_shapes

:4@*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	*
dtype0

&serving_default_bidirectional_21_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ4

StatefulPartitionedCallStatefulPartitionedCall&serving_default_bidirectional_21_input@bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel>bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/biasJbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernelAbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel?bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/biasKbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kerneldense_21/kerneldense_21/bias*
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
%__inference_signature_wrapper_5009287

NoOpNoOp
ûL
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¶L
value¬LB©L B¢L
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
VARIABLE_VALUEdense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE@bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEKbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¥
VARIABLE_VALUEHAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¯¨
VARIABLE_VALUERAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
£
VARIABLE_VALUEFAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¥
VARIABLE_VALUEHAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¯¨
VARIABLE_VALUERAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
£
VARIABLE_VALUEFAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
æ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOpTbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/Read/ReadVariableOp^bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/Read/ReadVariableOpRbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/Read/ReadVariableOpUbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/Read/ReadVariableOp_bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/Read/ReadVariableOpSbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp[Adam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/m/Read/ReadVariableOpeAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/m/Read/ReadVariableOpYAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/m/Read/ReadVariableOp\Adam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/m/Read/ReadVariableOpfAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/m/Read/ReadVariableOpZAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp[Adam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/v/Read/ReadVariableOpeAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/v/Read/ReadVariableOpYAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/v/Read/ReadVariableOp\Adam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/v/Read/ReadVariableOpfAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/v/Read/ReadVariableOpZAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/v/Read/ReadVariableOpConst*.
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
 __inference__traced_save_5011973
Í
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/bias@bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernelJbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel>bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/biasAbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernelKbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel?bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_21/kernel/mAdam/dense_21/bias/mGAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/mQAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/mEAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/mHAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/mRAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/mFAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/mAdam/dense_21/kernel/vAdam/dense_21/bias/vGAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/vQAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/vEAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/vHAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/vRAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/vFAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/v*-
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
#__inference__traced_restore_5012082·õ,
Ú	
Ã
/__inference_sequential_21_layer_call_fn_5009308

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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5008846o
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
û>
Í
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5010905
inputs_0C
1simple_rnn_cell_31_matmul_readvariableop_resource:4@@
2simple_rnn_cell_31_biasadd_readvariableop_resource:@E
3simple_rnn_cell_31_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_31/BiasAdd/ReadVariableOp¢(simple_rnn_cell_31/MatMul/ReadVariableOp¢*simple_rnn_cell_31/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_31/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_31/BiasAddBiasAdd#simple_rnn_cell_31/MatMul:product:01simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_31/MatMul_1MatMulzeros:output:02simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_31/addAddV2#simple_rnn_cell_31/BiasAdd:output:0%simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_31/TanhTanhsimple_rnn_cell_31/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_31_matmul_readvariableop_resource2simple_rnn_cell_31_biasadd_readvariableop_resource3simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
while_body_5010838*
condR
while_cond_5010837*8
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
NoOpNoOp*^simple_rnn_cell_31/BiasAdd/ReadVariableOp)^simple_rnn_cell_31/MatMul/ReadVariableOp+^simple_rnn_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_31/BiasAdd/ReadVariableOp)simple_rnn_cell_31/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_31/MatMul/ReadVariableOp(simple_rnn_cell_31/MatMul/ReadVariableOp2X
*simple_rnn_cell_31/MatMul_1/ReadVariableOp*simple_rnn_cell_31/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ØC

)backward_simple_rnn_10_while_body_5010222J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2I
Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_10_while_identity+
'backward_simple_rnn_10_while_identity_1+
'backward_simple_rnn_10_while_identity_2+
'backward_simple_rnn_10_while_identity_3+
'backward_simple_rnn_10_while_identity_4G
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp¢Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ£
@backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_10_while_placeholderWbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ö
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_10/while/simple_rnn_cell_32/MatMulMatMulGbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAddBiasAdd@backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul:product:0Nbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1MatMul*backward_simple_rnn_10_while_placeholder_2Obackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_10/while/simple_rnn_cell_32/addAddV2@backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd:output:0Bbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_10/while/simple_rnn_cell_32/TanhTanh7backward_simple_rnn_10/while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_10_while_placeholder_1Pbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_10/while/addAddV2(backward_simple_rnn_10_while_placeholder+backward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_10/while/add_1AddV2Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counter-backward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_10/while/IdentityIdentity&backward_simple_rnn_10/while/add_1:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_10/while/Identity_1IdentityLbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_10/while/Identity_2Identity$backward_simple_rnn_10/while/add:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_10/while/Identity_3IdentityQbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_10/while/Identity_4Identity8backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0"^backward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_10/while/NoOpNoOpG^backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpF^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpH^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0"W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0"[
'backward_simple_rnn_10_while_identity_10backward_simple_rnn_10/while/Identity_1:output:0"[
'backward_simple_rnn_10_while_identity_20backward_simple_rnn_10/while/Identity_2:output:0"[
'backward_simple_rnn_10_while_identity_30backward_simple_rnn_10/while/Identity_3:output:0"[
'backward_simple_rnn_10_while_identity_40backward_simple_rnn_10/while/Identity_4:output:0"¤
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resourceQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resourceRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resourcePbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0"
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpFbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpEbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp2
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpGbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
while_cond_5008191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5008191___redundant_placeholder05
1while_while_cond_5008191___redundant_placeholder15
1while_while_cond_5008191___redundant_placeholder25
1while_while_cond_5008191___redundant_placeholder3
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
¥

÷
E__inference_dense_21_layer_call_and_return_conditional_losses_5010751

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
»
Â
8__inference_backward_simple_rnn_10_layer_call_fn_5011268

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallø
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
GPU2*0J 8 *\
fWRU
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5008259o
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
?
Ë
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011235

inputsC
1simple_rnn_cell_31_matmul_readvariableop_resource:4@@
2simple_rnn_cell_31_biasadd_readvariableop_resource:@E
3simple_rnn_cell_31_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_31/BiasAdd/ReadVariableOp¢(simple_rnn_cell_31/MatMul/ReadVariableOp¢*simple_rnn_cell_31/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_31/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_31/BiasAddBiasAdd#simple_rnn_cell_31/MatMul:product:01simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_31/MatMul_1MatMulzeros:output:02simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_31/addAddV2#simple_rnn_cell_31/BiasAdd:output:0%simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_31/TanhTanhsimple_rnn_cell_31/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_31_matmul_readvariableop_resource2simple_rnn_cell_31_biasadd_readvariableop_resource3simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
while_body_5011168*
condR
while_cond_5011167*8
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
NoOpNoOp*^simple_rnn_cell_31/BiasAdd/ReadVariableOp)^simple_rnn_cell_31/MatMul/ReadVariableOp+^simple_rnn_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_31/BiasAdd/ReadVariableOp)simple_rnn_cell_31/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_31/MatMul/ReadVariableOp(simple_rnn_cell_31/MatMul/ReadVariableOp2X
*simple_rnn_cell_31/MatMul_1/ReadVariableOp*simple_rnn_cell_31/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
Ä
8__inference_backward_simple_rnn_10_layer_call_fn_5011257
inputs_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallú
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
GPU2*0J 8 *\
fWRU
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5008015o
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
ÑR
è
9bidirectional_21_forward_simple_rnn_10_while_body_5009372j
fbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_loop_counterp
lbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_maximum_iterations<
8bidirectional_21_forward_simple_rnn_10_while_placeholder>
:bidirectional_21_forward_simple_rnn_10_while_placeholder_1>
:bidirectional_21_forward_simple_rnn_10_while_placeholder_2i
ebidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_strided_slice_1_0¦
¡bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0r
`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@o
abidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@t
bbidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@9
5bidirectional_21_forward_simple_rnn_10_while_identity;
7bidirectional_21_forward_simple_rnn_10_while_identity_1;
7bidirectional_21_forward_simple_rnn_10_while_identity_2;
7bidirectional_21_forward_simple_rnn_10_while_identity_3;
7bidirectional_21_forward_simple_rnn_10_while_identity_4g
cbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_strided_slice_1¤
bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorp
^bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource:4@m
_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource:@r
`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢Vbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Ubidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp¢Wbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp¯
^bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ê
Pbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¡bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_08bidirectional_21_forward_simple_rnn_10_while_placeholdergbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ö
Ubidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0º
Fbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMulMatMulWbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0]bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
Vbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpabidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¶
Gbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAddBiasAddPbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul:product:0^bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
Wbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpbbidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¡
Hbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1MatMul:bidirectional_21_forward_simple_rnn_10_while_placeholder_2_bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Cbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/addAddV2Pbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd:output:0Rbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
Dbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/TanhTanhGbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Wbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Qbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem:bidirectional_21_forward_simple_rnn_10_while_placeholder_1`bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:0Hbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒt
2bidirectional_21/forward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ñ
0bidirectional_21/forward_simple_rnn_10/while/addAddV28bidirectional_21_forward_simple_rnn_10_while_placeholder;bidirectional_21/forward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: v
4bidirectional_21/forward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
2bidirectional_21/forward_simple_rnn_10/while/add_1AddV2fbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_loop_counter=bidirectional_21/forward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: Î
5bidirectional_21/forward_simple_rnn_10/while/IdentityIdentity6bidirectional_21/forward_simple_rnn_10/while/add_1:z:02^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
7bidirectional_21/forward_simple_rnn_10/while/Identity_1Identitylbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_maximum_iterations2^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Î
7bidirectional_21/forward_simple_rnn_10/while/Identity_2Identity4bidirectional_21/forward_simple_rnn_10/while/add:z:02^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: û
7bidirectional_21/forward_simple_rnn_10/while/Identity_3Identityabidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ó
7bidirectional_21/forward_simple_rnn_10/while/Identity_4IdentityHbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:02^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@þ
1bidirectional_21/forward_simple_rnn_10/while/NoOpNoOpW^bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpV^bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpX^bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ì
cbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_strided_slice_1ebidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_strided_slice_1_0"w
5bidirectional_21_forward_simple_rnn_10_while_identity>bidirectional_21/forward_simple_rnn_10/while/Identity:output:0"{
7bidirectional_21_forward_simple_rnn_10_while_identity_1@bidirectional_21/forward_simple_rnn_10/while/Identity_1:output:0"{
7bidirectional_21_forward_simple_rnn_10_while_identity_2@bidirectional_21/forward_simple_rnn_10/while/Identity_2:output:0"{
7bidirectional_21_forward_simple_rnn_10_while_identity_3@bidirectional_21/forward_simple_rnn_10/while/Identity_3:output:0"{
7bidirectional_21_forward_simple_rnn_10_while_identity_4@bidirectional_21/forward_simple_rnn_10/while/Identity_4:output:0"Ä
_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resourceabidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"Æ
`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resourcebbidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0"Â
^bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0"Æ
bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor¡bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2°
Vbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpVbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2®
Ubidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpUbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp2²
Wbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpWbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5011834

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
 5
¬
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5007715

inputs,
simple_rnn_cell_31_5007638:4@(
simple_rnn_cell_31_5007640:@,
simple_rnn_cell_31_5007642:@@
identity¢*simple_rnn_cell_31/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_31_5007638simple_rnn_cell_31_5007640simple_rnn_cell_31_5007642*
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5007598n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_31_5007638simple_rnn_cell_31_5007640simple_rnn_cell_31_5007642*
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
while_body_5007651*
condR
while_cond_5007650*8
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
NoOpNoOp+^simple_rnn_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_31/StatefulPartitionedCall*simple_rnn_cell_31/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
Ê

*__inference_dense_21_layer_call_fn_5010740

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
E__inference_dense_21_layer_call_and_return_conditional_losses_5008839o
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
þ¨
Û
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010731

inputsY
Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp¢Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp¢backward_simple_rnn_10/while¢?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp¢@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp¢forward_simple_rnn_10/whileQ
forward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:s
)forward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_10/strided_sliceStridedSlice$forward_simple_rnn_10/Shape:output:02forward_simple_rnn_10/strided_slice/stack:output:04forward_simple_rnn_10/strided_slice/stack_1:output:04forward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_10/zeros/packedPack,forward_simple_rnn_10/strided_slice:output:0-forward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_10/zerosFill+forward_simple_rnn_10/zeros/packed:output:0*forward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_10/transpose	Transposeinputs-forward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
forward_simple_rnn_10/Shape_1Shape#forward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_10/strided_slice_1StridedSlice&forward_simple_rnn_10/Shape_1:output:04forward_simple_rnn_10/strided_slice_1/stack:output:06forward_simple_rnn_10/strided_slice_1/stack_1:output:06forward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_10/TensorArrayV2TensorListReserve:forward_simple_rnn_10/TensorArrayV2/element_shape:output:0.forward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¢
=forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_10/transpose:y:0Tforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%forward_simple_rnn_10/strided_slice_2StridedSlice#forward_simple_rnn_10/transpose:y:04forward_simple_rnn_10/strided_slice_2/stack:output:06forward_simple_rnn_10/strided_slice_2/stack_1:output:06forward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_10/simple_rnn_cell_31/MatMulMatMul.forward_simple_rnn_10/strided_slice_2:output:0Fforward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_10/simple_rnn_cell_31/BiasAddBiasAdd9forward_simple_rnn_10/simple_rnn_cell_31/MatMul:product:0Gforward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1MatMul$forward_simple_rnn_10/zeros:output:0Hforward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_10/simple_rnn_cell_31/addAddV29forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd:output:0;forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_10/simple_rnn_cell_31/TanhTanh0forward_simple_rnn_10/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_10/TensorArrayV2_1TensorListReserve<forward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_10/whileWhile1forward_simple_rnn_10/while/loop_counter:output:07forward_simple_rnn_10/while/maximum_iterations:output:0#forward_simple_rnn_10/time:output:0.forward_simple_rnn_10/TensorArrayV2_1:handle:0$forward_simple_rnn_10/zeros:output:0.forward_simple_rnn_10/strided_slice_1:output:0Mforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resourceHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resourceIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_10_while_body_5010554*4
cond,R*
(forward_simple_rnn_10_while_cond_5010553*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_10/while:output:3Oforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_10/strided_slice_3StridedSliceAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_10/strided_slice_3/stack:output:06forward_simple_rnn_10/strided_slice_3/stack_1:output:06forward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_10/transpose_1	TransposeAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
backward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:t
*backward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_10/strided_sliceStridedSlice%backward_simple_rnn_10/Shape:output:03backward_simple_rnn_10/strided_slice/stack:output:05backward_simple_rnn_10/strided_slice/stack_1:output:05backward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_10/zeros/packedPack-backward_simple_rnn_10/strided_slice:output:0.backward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_10/zerosFill,backward_simple_rnn_10/zeros/packed:output:0+backward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
 backward_simple_rnn_10/transpose	Transposeinputs.backward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4r
backward_simple_rnn_10/Shape_1Shape$backward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_10/strided_slice_1StridedSlice'backward_simple_rnn_10/Shape_1:output:05backward_simple_rnn_10/strided_slice_1/stack:output:07backward_simple_rnn_10/strided_slice_1/stack_1:output:07backward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_10/TensorArrayV2TensorListReserve;backward_simple_rnn_10/TensorArrayV2/element_shape:output:0/backward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_10/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¹
 backward_simple_rnn_10/ReverseV2	ReverseV2$backward_simple_rnn_10/transpose:y:0.backward_simple_rnn_10/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Lbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ª
>backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_10/ReverseV2:output:0Ubackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
&backward_simple_rnn_10/strided_slice_2StridedSlice$backward_simple_rnn_10/transpose:y:05backward_simple_rnn_10/strided_slice_2/stack:output:07backward_simple_rnn_10/strided_slice_2/stack_1:output:07backward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÈ
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_10/simple_rnn_cell_32/MatMulMatMul/backward_simple_rnn_10/strided_slice_2:output:0Gbackward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_10/simple_rnn_cell_32/BiasAddBiasAdd:backward_simple_rnn_10/simple_rnn_cell_32/MatMul:product:0Hbackward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1MatMul%backward_simple_rnn_10/zeros:output:0Ibackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_10/simple_rnn_cell_32/addAddV2:backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd:output:0<backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_10/simple_rnn_cell_32/TanhTanh1backward_simple_rnn_10/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_10/TensorArrayV2_1TensorListReserve=backward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_10/whileWhile2backward_simple_rnn_10/while/loop_counter:output:08backward_simple_rnn_10/while/maximum_iterations:output:0$backward_simple_rnn_10/time:output:0/backward_simple_rnn_10/TensorArrayV2_1:handle:0%backward_simple_rnn_10/zeros:output:0/backward_simple_rnn_10/strided_slice_1:output:0Nbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resourceIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resourceJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *5
body-R+
)backward_simple_rnn_10_while_body_5010662*5
cond-R+
)backward_simple_rnn_10_while_cond_5010661*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_10/while:output:3Pbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_10/strided_slice_3StridedSliceBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_10/strided_slice_3/stack:output:07backward_simple_rnn_10/strided_slice_3/stack_1:output:07backward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_10/transpose_1	TransposeBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_10/strided_slice_3:output:0/backward_simple_rnn_10/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@^backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpB^backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp^backward_simple_rnn_10/while@^forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?^forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpA^forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp^forward_simple_rnn_10/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp2
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpAbackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp2<
backward_simple_rnn_10/whilebackward_simple_rnn_10/while2
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp2
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp2:
forward_simple_rnn_10/whileforward_simple_rnn_10/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
?
Ë
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5008140

inputsC
1simple_rnn_cell_31_matmul_readvariableop_resource:4@@
2simple_rnn_cell_31_biasadd_readvariableop_resource:@E
3simple_rnn_cell_31_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_31/BiasAdd/ReadVariableOp¢(simple_rnn_cell_31/MatMul/ReadVariableOp¢*simple_rnn_cell_31/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_31/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_31/BiasAddBiasAdd#simple_rnn_cell_31/MatMul:product:01simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_31/MatMul_1MatMulzeros:output:02simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_31/addAddV2#simple_rnn_cell_31/BiasAdd:output:0%simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_31/TanhTanhsimple_rnn_cell_31/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_31_matmul_readvariableop_resource2simple_rnn_cell_31_biasadd_readvariableop_resource3simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
while_body_5008073*
condR
while_cond_5008072*8
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
NoOpNoOp*^simple_rnn_cell_31/BiasAdd/ReadVariableOp)^simple_rnn_cell_31/MatMul/ReadVariableOp+^simple_rnn_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_31/BiasAdd/ReadVariableOp)simple_rnn_cell_31/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_31/MatMul/ReadVariableOp(simple_rnn_cell_31/MatMul/ReadVariableOp2X
*simple_rnn_cell_31/MatMul_1/ReadVariableOp*simple_rnn_cell_31/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
Ñ
(forward_simple_rnn_10_while_cond_5010553H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2J
Fforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010553___redundant_placeholder0a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010553___redundant_placeholder1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010553___redundant_placeholder2a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010553___redundant_placeholder3(
$forward_simple_rnn_10_while_identity
º
 forward_simple_rnn_10/while/LessLess'forward_simple_rnn_10_while_placeholderFforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_10/while/IdentityIdentity$forward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0*(
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5011789

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
	

2__inference_bidirectional_21_layer_call_fn_5009834

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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5008814p
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5007896

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
2__inference_bidirectional_21_layer_call_fn_5009817
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5008573p
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
Ú@
Î
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011391
inputs_0C
1simple_rnn_cell_32_matmul_readvariableop_resource:4@@
2simple_rnn_cell_32_biasadd_readvariableop_resource:@E
3simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_32/BiasAdd/ReadVariableOp¢(simple_rnn_cell_32/MatMul/ReadVariableOp¢*simple_rnn_cell_32/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_32/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_32/BiasAddBiasAdd#simple_rnn_cell_32/MatMul:product:01simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_32/MatMul_1MatMulzeros:output:02simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_32/addAddV2#simple_rnn_cell_32/BiasAdd:output:0%simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_32/TanhTanhsimple_rnn_cell_32/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_32_matmul_readvariableop_resource2simple_rnn_cell_32_biasadd_readvariableop_resource3simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
while_body_5011324*
condR
while_cond_5011323*8
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
NoOpNoOp*^simple_rnn_cell_32/BiasAdd/ReadVariableOp)^simple_rnn_cell_32/MatMul/ReadVariableOp+^simple_rnn_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_32/BiasAdd/ReadVariableOp)simple_rnn_cell_32/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_32/MatMul/ReadVariableOp(simple_rnn_cell_32/MatMul/ReadVariableOp2X
*simple_rnn_cell_32/MatMul_1/ReadVariableOp*simple_rnn_cell_32/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ÉS

:bidirectional_21_backward_simple_rnn_10_while_body_5009480l
hbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_loop_counterr
nbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_maximum_iterations=
9bidirectional_21_backward_simple_rnn_10_while_placeholder?
;bidirectional_21_backward_simple_rnn_10_while_placeholder_1?
;bidirectional_21_backward_simple_rnn_10_while_placeholder_2k
gbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_strided_slice_1_0¨
£bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0s
abidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@p
bbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@u
cbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@:
6bidirectional_21_backward_simple_rnn_10_while_identity<
8bidirectional_21_backward_simple_rnn_10_while_identity_1<
8bidirectional_21_backward_simple_rnn_10_while_identity_2<
8bidirectional_21_backward_simple_rnn_10_while_identity_3<
8bidirectional_21_backward_simple_rnn_10_while_identity_4i
ebidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_strided_slice_1¦
¡bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorq
_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource:4@n
`bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource:@s
abidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢Wbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Vbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp¢Xbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp°
_bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ï
Qbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem£bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_09bidirectional_21_backward_simple_rnn_10_while_placeholderhbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ø
Vbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpabidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0½
Gbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMulMatMulXbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0^bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ö
Wbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpbbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¹
Hbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAddBiasAddQbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul:product:0_bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ü
Xbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpcbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¤
Ibidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1MatMul;bidirectional_21_backward_simple_rnn_10_while_placeholder_2`bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
Dbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/addAddV2Qbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd:output:0Sbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@É
Ebidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/TanhTanhHbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Xbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Rbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem;bidirectional_21_backward_simple_rnn_10_while_placeholder_1abidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:0Ibidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒu
3bidirectional_21/backward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ô
1bidirectional_21/backward_simple_rnn_10/while/addAddV29bidirectional_21_backward_simple_rnn_10_while_placeholder<bidirectional_21/backward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: w
5bidirectional_21/backward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
3bidirectional_21/backward_simple_rnn_10/while/add_1AddV2hbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_loop_counter>bidirectional_21/backward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: Ñ
6bidirectional_21/backward_simple_rnn_10/while/IdentityIdentity7bidirectional_21/backward_simple_rnn_10/while/add_1:z:03^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
8bidirectional_21/backward_simple_rnn_10/while/Identity_1Identitynbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_maximum_iterations3^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ñ
8bidirectional_21/backward_simple_rnn_10/while/Identity_2Identity5bidirectional_21/backward_simple_rnn_10/while/add:z:03^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: þ
8bidirectional_21/backward_simple_rnn_10/while/Identity_3Identitybbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ö
8bidirectional_21/backward_simple_rnn_10/while/Identity_4IdentityIbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:03^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2bidirectional_21/backward_simple_rnn_10/while/NoOpNoOpX^bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpW^bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpY^bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ð
ebidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_strided_slice_1gbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_strided_slice_1_0"y
6bidirectional_21_backward_simple_rnn_10_while_identity?bidirectional_21/backward_simple_rnn_10/while/Identity:output:0"}
8bidirectional_21_backward_simple_rnn_10_while_identity_1Abidirectional_21/backward_simple_rnn_10/while/Identity_1:output:0"}
8bidirectional_21_backward_simple_rnn_10_while_identity_2Abidirectional_21/backward_simple_rnn_10/while/Identity_2:output:0"}
8bidirectional_21_backward_simple_rnn_10_while_identity_3Abidirectional_21/backward_simple_rnn_10/while/Identity_3:output:0"}
8bidirectional_21_backward_simple_rnn_10_while_identity_4Abidirectional_21/backward_simple_rnn_10/while/Identity_4:output:0"Æ
`bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resourcebbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"È
abidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resourcecbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"Ä
_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resourceabidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0"Ê
¡bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor£bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2²
Wbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpWbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2°
Vbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpVbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp2´
Xbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpXbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
while_body_5007651
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_31_5007673_0:4@0
"while_simple_rnn_cell_31_5007675_0:@4
"while_simple_rnn_cell_31_5007677_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_31_5007673:4@.
 while_simple_rnn_cell_31_5007675:@2
 while_simple_rnn_cell_31_5007677:@@¢0while/simple_rnn_cell_31/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_31_5007673_0"while_simple_rnn_cell_31_5007675_0"while_simple_rnn_cell_31_5007677_0*
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5007598r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_31/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_31_5007673"while_simple_rnn_cell_31_5007673_0"F
 while_simple_rnn_cell_31_5007675"while_simple_rnn_cell_31_5007675_0"F
 while_simple_rnn_cell_31_5007677"while_simple_rnn_cell_31_5007677_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_31/StatefulPartitionedCall0while/simple_rnn_cell_31/StatefulPartitionedCall: 
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


#__inference__traced_restore_5012082
file_prefix3
 assignvariableop_dense_21_kernel:	.
 assignvariableop_1_dense_21_bias:e
Sassignvariableop_2_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel:4@o
]assignvariableop_3_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel:@@_
Qassignvariableop_4_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias:@f
Tassignvariableop_5_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel:4@p
^assignvariableop_6_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel:@@`
Rassignvariableop_7_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias:@&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: =
*assignvariableop_17_adam_dense_21_kernel_m:	6
(assignvariableop_18_adam_dense_21_bias_m:m
[assignvariableop_19_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_m:4@w
eassignvariableop_20_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_m:@@g
Yassignvariableop_21_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_m:@n
\assignvariableop_22_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_m:4@x
fassignvariableop_23_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_m:@@h
Zassignvariableop_24_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_m:@=
*assignvariableop_25_adam_dense_21_kernel_v:	6
(assignvariableop_26_adam_dense_21_bias_v:m
[assignvariableop_27_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_v:4@w
eassignvariableop_28_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_v:@@g
Yassignvariableop_29_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_v:@n
\assignvariableop_30_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_v:4@x
fassignvariableop_31_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_v:@@h
Zassignvariableop_32_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_v:@
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
AssignVariableOpAssignVariableOp assignvariableop_dense_21_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_21_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_2AssignVariableOpSassignvariableop_2_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_3AssignVariableOp]assignvariableop_3_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_4AssignVariableOpQassignvariableop_4_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_5AssignVariableOpTassignvariableop_5_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_6AssignVariableOp^assignvariableop_6_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_7AssignVariableOpRassignvariableop_7_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_21_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_21_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_19AssignVariableOp[assignvariableop_19_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_20AssignVariableOpeassignvariableop_20_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_21AssignVariableOpYassignvariableop_21_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_22AssignVariableOp\assignvariableop_22_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:×
AssignVariableOp_23AssignVariableOpfassignvariableop_23_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_24AssignVariableOpZassignvariableop_24_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_21_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_21_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_27AssignVariableOp[assignvariableop_27_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_28AssignVariableOpeassignvariableop_28_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_29AssignVariableOpYassignvariableop_29_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_30AssignVariableOp\assignvariableop_30_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:×
AssignVariableOp_31AssignVariableOpfassignvariableop_31_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_32AssignVariableOpZassignvariableop_32_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_vIdentity_32:output:0"/device:CPU:0*
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
ÿ
ê
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5007476

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
ÉS

:bidirectional_21_backward_simple_rnn_10_while_body_5009707l
hbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_loop_counterr
nbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_maximum_iterations=
9bidirectional_21_backward_simple_rnn_10_while_placeholder?
;bidirectional_21_backward_simple_rnn_10_while_placeholder_1?
;bidirectional_21_backward_simple_rnn_10_while_placeholder_2k
gbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_strided_slice_1_0¨
£bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0s
abidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@p
bbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@u
cbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@:
6bidirectional_21_backward_simple_rnn_10_while_identity<
8bidirectional_21_backward_simple_rnn_10_while_identity_1<
8bidirectional_21_backward_simple_rnn_10_while_identity_2<
8bidirectional_21_backward_simple_rnn_10_while_identity_3<
8bidirectional_21_backward_simple_rnn_10_while_identity_4i
ebidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_strided_slice_1¦
¡bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorq
_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource:4@n
`bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource:@s
abidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢Wbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Vbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp¢Xbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp°
_bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ï
Qbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem£bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_09bidirectional_21_backward_simple_rnn_10_while_placeholderhbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ø
Vbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpabidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0½
Gbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMulMatMulXbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0^bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ö
Wbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpbbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¹
Hbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAddBiasAddQbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul:product:0_bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ü
Xbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpcbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¤
Ibidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1MatMul;bidirectional_21_backward_simple_rnn_10_while_placeholder_2`bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
Dbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/addAddV2Qbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd:output:0Sbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@É
Ebidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/TanhTanhHbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Xbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Rbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem;bidirectional_21_backward_simple_rnn_10_while_placeholder_1abidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:0Ibidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒu
3bidirectional_21/backward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ô
1bidirectional_21/backward_simple_rnn_10/while/addAddV29bidirectional_21_backward_simple_rnn_10_while_placeholder<bidirectional_21/backward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: w
5bidirectional_21/backward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
3bidirectional_21/backward_simple_rnn_10/while/add_1AddV2hbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_loop_counter>bidirectional_21/backward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: Ñ
6bidirectional_21/backward_simple_rnn_10/while/IdentityIdentity7bidirectional_21/backward_simple_rnn_10/while/add_1:z:03^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
8bidirectional_21/backward_simple_rnn_10/while/Identity_1Identitynbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_maximum_iterations3^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ñ
8bidirectional_21/backward_simple_rnn_10/while/Identity_2Identity5bidirectional_21/backward_simple_rnn_10/while/add:z:03^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: þ
8bidirectional_21/backward_simple_rnn_10/while/Identity_3Identitybbidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ö
8bidirectional_21/backward_simple_rnn_10/while/Identity_4IdentityIbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:03^bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2bidirectional_21/backward_simple_rnn_10/while/NoOpNoOpX^bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpW^bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpY^bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ð
ebidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_strided_slice_1gbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_strided_slice_1_0"y
6bidirectional_21_backward_simple_rnn_10_while_identity?bidirectional_21/backward_simple_rnn_10/while/Identity:output:0"}
8bidirectional_21_backward_simple_rnn_10_while_identity_1Abidirectional_21/backward_simple_rnn_10/while/Identity_1:output:0"}
8bidirectional_21_backward_simple_rnn_10_while_identity_2Abidirectional_21/backward_simple_rnn_10/while/Identity_2:output:0"}
8bidirectional_21_backward_simple_rnn_10_while_identity_3Abidirectional_21/backward_simple_rnn_10/while/Identity_3:output:0"}
8bidirectional_21_backward_simple_rnn_10_while_identity_4Abidirectional_21/backward_simple_rnn_10/while/Identity_4:output:0"Æ
`bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resourcebbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"È
abidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resourcecbidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"Ä
_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resourceabidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0"Ê
¡bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor£bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2²
Wbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpWbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2°
Vbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpVbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp2´
Xbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpXbidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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


Ó
/__inference_sequential_21_layer_call_fn_5008865
bidirectional_21_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5008846o
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
_user_specified_namebidirectional_21_input
²
Ñ
(forward_simple_rnn_10_while_cond_5010113H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2J
Fforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010113___redundant_placeholder0a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010113___redundant_placeholder1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010113___redundant_placeholder2a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010113___redundant_placeholder3(
$forward_simple_rnn_10_while_identity
º
 forward_simple_rnn_10/while/LessLess'forward_simple_rnn_10_while_placeholderFforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_10/while/IdentityIdentity$forward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0*(
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
A
Ì
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5008259

inputsC
1simple_rnn_cell_32_matmul_readvariableop_resource:4@@
2simple_rnn_cell_32_biasadd_readvariableop_resource:@E
3simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_32/BiasAdd/ReadVariableOp¢(simple_rnn_cell_32/MatMul/ReadVariableOp¢*simple_rnn_cell_32/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_32/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_32/BiasAddBiasAdd#simple_rnn_cell_32/MatMul:product:01simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_32/MatMul_1MatMulzeros:output:02simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_32/addAddV2#simple_rnn_cell_32/BiasAdd:output:0%simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_32/TanhTanhsimple_rnn_cell_32/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_32_matmul_readvariableop_resource2simple_rnn_cell_32_biasadd_readvariableop_resource3simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
while_body_5008192*
condR
while_cond_5008191*8
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
NoOpNoOp*^simple_rnn_cell_32/BiasAdd/ReadVariableOp)^simple_rnn_cell_32/MatMul/ReadVariableOp+^simple_rnn_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_32/BiasAdd/ReadVariableOp)simple_rnn_cell_32/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_32/MatMul/ReadVariableOp(simple_rnn_cell_32/MatMul/ReadVariableOp2X
*simple_rnn_cell_32/MatMul_1/ReadVariableOp*simple_rnn_cell_32/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü-
Ò
while_body_5011058
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_31_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_31/MatMul/ReadVariableOp¢0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_31/BiasAddBiasAdd)while/simple_rnn_cell_31/MatMul:product:07while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_31/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_31/addAddV2)while/simple_rnn_cell_31/BiasAdd:output:0+while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_31/TanhTanh while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_31/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_31/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_31/MatMul/ReadVariableOp1^while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_31_matmul_readvariableop_resource9while_simple_rnn_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_31/MatMul/ReadVariableOp.while/simple_rnn_cell_31/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
£
¿
Hsequential_21_bidirectional_21_backward_simple_rnn_10_while_cond_5007351
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_loop_counter
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_maximum_iterationsK
Gsequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholderM
Isequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholder_1M
Isequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholder_2
sequential_21_bidirectional_21_backward_simple_rnn_10_while_less_sequential_21_bidirectional_21_backward_simple_rnn_10_strided_slice_1¢
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_cond_5007351___redundant_placeholder0¢
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_cond_5007351___redundant_placeholder1¢
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_cond_5007351___redundant_placeholder2¢
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_cond_5007351___redundant_placeholder3H
Dsequential_21_bidirectional_21_backward_simple_rnn_10_while_identity
»
@sequential_21/bidirectional_21/backward_simple_rnn_10/while/LessLessGsequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholdersequential_21_bidirectional_21_backward_simple_rnn_10_while_less_sequential_21_bidirectional_21_backward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: ·
Dsequential_21/bidirectional_21/backward_simple_rnn_10/while/IdentityIdentityDsequential_21/bidirectional_21/backward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "
Dsequential_21_bidirectional_21_backward_simple_rnn_10_while_identityMsequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity:output:0*(
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
while_body_5010948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_31_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_31/MatMul/ReadVariableOp¢0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_31/BiasAddBiasAdd)while/simple_rnn_cell_31/MatMul:product:07while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_31/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_31/addAddV2)while/simple_rnn_cell_31/BiasAdd:output:0+while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_31/TanhTanh while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_31/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_31/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_31/MatMul/ReadVariableOp1^while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_31_matmul_readvariableop_resource9while_simple_rnn_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_31/MatMul/ReadVariableOp.while/simple_rnn_cell_31/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
while_cond_5011547
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5011547___redundant_placeholder05
1while_while_cond_5011547___redundant_placeholder15
1while_while_cond_5011547___redundant_placeholder25
1while_while_cond_5011547___redundant_placeholder3
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
(forward_simple_rnn_10_while_body_5010554H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2G
Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0
forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_10_while_identity*
&forward_simple_rnn_10_while_identity_1*
&forward_simple_rnn_10_while_identity_2*
&forward_simple_rnn_10_while_identity_3*
&forward_simple_rnn_10_while_identity_4E
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp¢Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
Mforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_10_while_placeholderVforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_10/while/simple_rnn_cell_31/MatMulMatMulFforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAddBiasAdd?forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul:product:0Mforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1MatMul)forward_simple_rnn_10_while_placeholder_2Nforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_10/while/simple_rnn_cell_31/addAddV2?forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd:output:0Aforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_10/while/simple_rnn_cell_31/TanhTanh6forward_simple_rnn_10/while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_10_while_placeholder_1Oforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_10/while/addAddV2'forward_simple_rnn_10_while_placeholder*forward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_10/while/add_1AddV2Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counter,forward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_10/while/IdentityIdentity%forward_simple_rnn_10/while/add_1:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_10/while/Identity_1IdentityJforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_10/while/Identity_2Identity#forward_simple_rnn_10/while/add:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_10/while/Identity_3IdentityPforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_10/while/Identity_4Identity7forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0!^forward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_10/while/NoOpNoOpF^forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpE^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpG^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0"U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0"Y
&forward_simple_rnn_10_while_identity_1/forward_simple_rnn_10/while/Identity_1:output:0"Y
&forward_simple_rnn_10_while_identity_2/forward_simple_rnn_10/while/Identity_2:output:0"Y
&forward_simple_rnn_10_while_identity_3/forward_simple_rnn_10/while/Identity_3:output:0"Y
&forward_simple_rnn_10_while_identity_4/forward_simple_rnn_10/while/Identity_4:output:0"¢
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resourcePforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resourceQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resourceOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0"
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpEforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpDforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp2
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpFforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
ó-
Ò
while_body_5010838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_31_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_31/MatMul/ReadVariableOp¢0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_31/BiasAddBiasAdd)while/simple_rnn_cell_31/MatMul:product:07while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_31/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_31/addAddV2)while/simple_rnn_cell_31/BiasAdd:output:0+while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_31/TanhTanh while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_31/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_31/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_31/MatMul/ReadVariableOp1^while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_31_matmul_readvariableop_resource9while_simple_rnn_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_31/MatMul/ReadVariableOp.while/simple_rnn_cell_31/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
while_cond_5007650
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5007650___redundant_placeholder05
1while_while_cond_5007650___redundant_placeholder15
1while_while_cond_5007650___redundant_placeholder25
1while_while_cond_5007650___redundant_placeholder3
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
(forward_simple_rnn_10_while_body_5010334H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2G
Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0
forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_10_while_identity*
&forward_simple_rnn_10_while_identity_1*
&forward_simple_rnn_10_while_identity_2*
&forward_simple_rnn_10_while_identity_3*
&forward_simple_rnn_10_while_identity_4E
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp¢Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
Mforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_10_while_placeholderVforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_10/while/simple_rnn_cell_31/MatMulMatMulFforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAddBiasAdd?forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul:product:0Mforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1MatMul)forward_simple_rnn_10_while_placeholder_2Nforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_10/while/simple_rnn_cell_31/addAddV2?forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd:output:0Aforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_10/while/simple_rnn_cell_31/TanhTanh6forward_simple_rnn_10/while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_10_while_placeholder_1Oforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_10/while/addAddV2'forward_simple_rnn_10_while_placeholder*forward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_10/while/add_1AddV2Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counter,forward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_10/while/IdentityIdentity%forward_simple_rnn_10/while/add_1:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_10/while/Identity_1IdentityJforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_10/while/Identity_2Identity#forward_simple_rnn_10/while/add:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_10/while/Identity_3IdentityPforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_10/while/Identity_4Identity7forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0!^forward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_10/while/NoOpNoOpF^forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpE^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpG^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0"U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0"Y
&forward_simple_rnn_10_while_identity_1/forward_simple_rnn_10/while/Identity_1:output:0"Y
&forward_simple_rnn_10_while_identity_2/forward_simple_rnn_10/while/Identity_2:output:0"Y
&forward_simple_rnn_10_while_identity_3/forward_simple_rnn_10/while/Identity_3:output:0"Y
&forward_simple_rnn_10_while_identity_4/forward_simple_rnn_10/while/Identity_4:output:0"¢
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resourcePforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resourceQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resourceOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0"
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpEforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpDforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp2
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpFforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
while_cond_5011057
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5011057___redundant_placeholder05
1while_while_cond_5011057___redundant_placeholder15
1while_while_cond_5011057___redundant_placeholder25
1while_while_cond_5011057___redundant_placeholder3
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
9bidirectional_21_forward_simple_rnn_10_while_body_5009599j
fbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_loop_counterp
lbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_maximum_iterations<
8bidirectional_21_forward_simple_rnn_10_while_placeholder>
:bidirectional_21_forward_simple_rnn_10_while_placeholder_1>
:bidirectional_21_forward_simple_rnn_10_while_placeholder_2i
ebidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_strided_slice_1_0¦
¡bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0r
`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@o
abidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@t
bbidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@9
5bidirectional_21_forward_simple_rnn_10_while_identity;
7bidirectional_21_forward_simple_rnn_10_while_identity_1;
7bidirectional_21_forward_simple_rnn_10_while_identity_2;
7bidirectional_21_forward_simple_rnn_10_while_identity_3;
7bidirectional_21_forward_simple_rnn_10_while_identity_4g
cbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_strided_slice_1¤
bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorp
^bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource:4@m
_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource:@r
`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢Vbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Ubidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp¢Wbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp¯
^bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ê
Pbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¡bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_08bidirectional_21_forward_simple_rnn_10_while_placeholdergbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ö
Ubidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0º
Fbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMulMatMulWbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0]bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
Vbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpabidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¶
Gbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAddBiasAddPbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul:product:0^bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
Wbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpbbidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¡
Hbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1MatMul:bidirectional_21_forward_simple_rnn_10_while_placeholder_2_bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Cbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/addAddV2Pbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd:output:0Rbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
Dbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/TanhTanhGbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Wbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Qbidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem:bidirectional_21_forward_simple_rnn_10_while_placeholder_1`bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:0Hbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒt
2bidirectional_21/forward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ñ
0bidirectional_21/forward_simple_rnn_10/while/addAddV28bidirectional_21_forward_simple_rnn_10_while_placeholder;bidirectional_21/forward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: v
4bidirectional_21/forward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
2bidirectional_21/forward_simple_rnn_10/while/add_1AddV2fbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_loop_counter=bidirectional_21/forward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: Î
5bidirectional_21/forward_simple_rnn_10/while/IdentityIdentity6bidirectional_21/forward_simple_rnn_10/while/add_1:z:02^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
7bidirectional_21/forward_simple_rnn_10/while/Identity_1Identitylbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_maximum_iterations2^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Î
7bidirectional_21/forward_simple_rnn_10/while/Identity_2Identity4bidirectional_21/forward_simple_rnn_10/while/add:z:02^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: û
7bidirectional_21/forward_simple_rnn_10/while/Identity_3Identityabidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ó
7bidirectional_21/forward_simple_rnn_10/while/Identity_4IdentityHbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:02^bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@þ
1bidirectional_21/forward_simple_rnn_10/while/NoOpNoOpW^bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpV^bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpX^bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ì
cbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_strided_slice_1ebidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_strided_slice_1_0"w
5bidirectional_21_forward_simple_rnn_10_while_identity>bidirectional_21/forward_simple_rnn_10/while/Identity:output:0"{
7bidirectional_21_forward_simple_rnn_10_while_identity_1@bidirectional_21/forward_simple_rnn_10/while/Identity_1:output:0"{
7bidirectional_21_forward_simple_rnn_10_while_identity_2@bidirectional_21/forward_simple_rnn_10/while/Identity_2:output:0"{
7bidirectional_21_forward_simple_rnn_10_while_identity_3@bidirectional_21/forward_simple_rnn_10/while/Identity_3:output:0"{
7bidirectional_21_forward_simple_rnn_10_while_identity_4@bidirectional_21/forward_simple_rnn_10/while/Identity_4:output:0"Ä
_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resourceabidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"Æ
`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resourcebbidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0"Â
^bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource`bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0"Æ
bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor¡bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2°
Vbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpVbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2®
Ubidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpUbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp2²
Wbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpWbidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
2__inference_bidirectional_21_layer_call_fn_5009851

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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5009114p
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
7__inference_forward_simple_rnn_10_layer_call_fn_5010795

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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5008542o
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
­
Ã
7__inference_forward_simple_rnn_10_layer_call_fn_5010773
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5007715o
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009174

inputs*
bidirectional_21_5009155:4@&
bidirectional_21_5009157:@*
bidirectional_21_5009159:@@*
bidirectional_21_5009161:4@&
bidirectional_21_5009163:@*
bidirectional_21_5009165:@@#
dense_21_5009168:	
dense_21_5009170:
identity¢(bidirectional_21/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall
(bidirectional_21/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_21_5009155bidirectional_21_5009157bidirectional_21_5009159bidirectional_21_5009161bidirectional_21_5009163bidirectional_21_5009165*
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5009114¡
 dense_21/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_21/StatefulPartitionedCall:output:0dense_21_5009168dense_21_5009170*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_5008839x
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_21/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_21/StatefulPartitionedCall(bidirectional_21/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
À

Ü
4__inference_simple_rnn_cell_31_layer_call_fn_5011741

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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5007476o
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
ÏC

)backward_simple_rnn_10_while_body_5009045J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2I
Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_10_while_identity+
'backward_simple_rnn_10_while_identity_1+
'backward_simple_rnn_10_while_identity_2+
'backward_simple_rnn_10_while_identity_3+
'backward_simple_rnn_10_while_identity_4G
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp¢Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
@backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_10_while_placeholderWbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ö
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_10/while/simple_rnn_cell_32/MatMulMatMulGbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAddBiasAdd@backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul:product:0Nbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1MatMul*backward_simple_rnn_10_while_placeholder_2Obackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_10/while/simple_rnn_cell_32/addAddV2@backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd:output:0Bbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_10/while/simple_rnn_cell_32/TanhTanh7backward_simple_rnn_10/while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_10_while_placeholder_1Pbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_10/while/addAddV2(backward_simple_rnn_10_while_placeholder+backward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_10/while/add_1AddV2Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counter-backward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_10/while/IdentityIdentity&backward_simple_rnn_10/while/add_1:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_10/while/Identity_1IdentityLbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_10/while/Identity_2Identity$backward_simple_rnn_10/while/add:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_10/while/Identity_3IdentityQbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_10/while/Identity_4Identity8backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0"^backward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_10/while/NoOpNoOpG^backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpF^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpH^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0"W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0"[
'backward_simple_rnn_10_while_identity_10backward_simple_rnn_10/while/Identity_1:output:0"[
'backward_simple_rnn_10_while_identity_20backward_simple_rnn_10/while/Identity_2:output:0"[
'backward_simple_rnn_10_while_identity_30backward_simple_rnn_10/while/Identity_3:output:0"[
'backward_simple_rnn_10_while_identity_40backward_simple_rnn_10/while/Identity_4:output:0"¤
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resourceQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resourceRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resourcePbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0"
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpFbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpEbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp2
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpGbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
while_cond_5010947
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5010947___redundant_placeholder05
1while_while_cond_5010947___redundant_placeholder15
1while_while_cond_5010947___redundant_placeholder25
1while_while_cond_5010947___redundant_placeholder3
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
while_body_5011436
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_32_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_32/MatMul/ReadVariableOp¢0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_32/BiasAddBiasAdd)while/simple_rnn_cell_32/MatMul:product:07while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_32/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_32/addAddV2)while/simple_rnn_cell_32/BiasAdd:output:0+while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_32/TanhTanh while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_32/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_32/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_32/MatMul/ReadVariableOp1^while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_32_matmul_readvariableop_resource9while_simple_rnn_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_32/MatMul/ReadVariableOp.while/simple_rnn_cell_32/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
while_body_5007951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_32_5007973_0:4@0
"while_simple_rnn_cell_32_5007975_0:@4
"while_simple_rnn_cell_32_5007977_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_32_5007973:4@.
 while_simple_rnn_cell_32_5007975:@2
 while_simple_rnn_cell_32_5007977:@@¢0while/simple_rnn_cell_32/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_32/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_32_5007973_0"while_simple_rnn_cell_32_5007975_0"while_simple_rnn_cell_32_5007977_0*
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5007896r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_32/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_32/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_32_5007973"while_simple_rnn_cell_32_5007973_0"F
 while_simple_rnn_cell_32_5007975"while_simple_rnn_cell_32_5007975_0"F
 while_simple_rnn_cell_32_5007977"while_simple_rnn_cell_32_5007977_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_32/StatefulPartitionedCall0while/simple_rnn_cell_32/StatefulPartitionedCall: 
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5007774

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
while_cond_5007950
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5007950___redundant_placeholder05
1while_while_cond_5007950___redundant_placeholder15
1while_while_cond_5007950___redundant_placeholder25
1while_while_cond_5007950___redundant_placeholder3
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5007598

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
while_cond_5008072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5008072___redundant_placeholder05
1while_while_cond_5008072___redundant_placeholder15
1while_while_cond_5008072___redundant_placeholder25
1while_while_cond_5008072___redundant_placeholder3
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
¡
¯	
:bidirectional_21_backward_simple_rnn_10_while_cond_5009706l
hbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_loop_counterr
nbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_maximum_iterations=
9bidirectional_21_backward_simple_rnn_10_while_placeholder?
;bidirectional_21_backward_simple_rnn_10_while_placeholder_1?
;bidirectional_21_backward_simple_rnn_10_while_placeholder_2n
jbidirectional_21_backward_simple_rnn_10_while_less_bidirectional_21_backward_simple_rnn_10_strided_slice_1
bidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_cond_5009706___redundant_placeholder0
bidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_cond_5009706___redundant_placeholder1
bidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_cond_5009706___redundant_placeholder2
bidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_cond_5009706___redundant_placeholder3:
6bidirectional_21_backward_simple_rnn_10_while_identity

2bidirectional_21/backward_simple_rnn_10/while/LessLess9bidirectional_21_backward_simple_rnn_10_while_placeholderjbidirectional_21_backward_simple_rnn_10_while_less_bidirectional_21_backward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: 
6bidirectional_21/backward_simple_rnn_10/while/IdentityIdentity6bidirectional_21/backward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "y
6bidirectional_21_backward_simple_rnn_10_while_identity?bidirectional_21/backward_simple_rnn_10/while/Identity:output:0*(
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
Í
ä
)backward_simple_rnn_10_while_cond_5010441J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2L
Hbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010441___redundant_placeholder0c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010441___redundant_placeholder1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010441___redundant_placeholder2c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010441___redundant_placeholder3)
%backward_simple_rnn_10_while_identity
¾
!backward_simple_rnn_10/while/LessLess(backward_simple_rnn_10_while_placeholderHbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_10/while/IdentityIdentity%backward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0*(
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


Ó
/__inference_sequential_21_layer_call_fn_5009214
bidirectional_21_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009174o
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
_user_specified_namebidirectional_21_input
ü-
Ò
while_body_5008192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_32_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_32/MatMul/ReadVariableOp¢0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_32/BiasAddBiasAdd)while/simple_rnn_cell_32/MatMul:product:07while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_32/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_32/addAddV2)while/simple_rnn_cell_32/BiasAdd:output:0+while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_32/TanhTanh while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_32/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_32/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_32/MatMul/ReadVariableOp1^while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_32_matmul_readvariableop_resource9while_simple_rnn_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_32/MatMul/ReadVariableOp.while/simple_rnn_cell_32/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
/__inference_sequential_21_layer_call_fn_5009329

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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009174o
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
ûñ
á
"__inference__wrapped_model_5007428
bidirectional_21_inputx
fsequential_21_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource:4@u
gsequential_21_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource:@z
hsequential_21_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@y
gsequential_21_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource:4@v
hsequential_21_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource:@{
isequential_21_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@H
5sequential_21_dense_21_matmul_readvariableop_resource:	D
6sequential_21_dense_21_biasadd_readvariableop_resource:
identity¢_sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢^sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp¢`sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp¢;sequential_21/bidirectional_21/backward_simple_rnn_10/while¢^sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢]sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp¢_sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp¢:sequential_21/bidirectional_21/forward_simple_rnn_10/while¢-sequential_21/dense_21/BiasAdd/ReadVariableOp¢,sequential_21/dense_21/MatMul/ReadVariableOp
:sequential_21/bidirectional_21/forward_simple_rnn_10/ShapeShapebidirectional_21_input*
T0*
_output_shapes
:
Hsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ú
Bsequential_21/bidirectional_21/forward_simple_rnn_10/strided_sliceStridedSliceCsequential_21/bidirectional_21/forward_simple_rnn_10/Shape:output:0Qsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice/stack:output:0Ssequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice/stack_1:output:0Ssequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Csequential_21/bidirectional_21/forward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
Asequential_21/bidirectional_21/forward_simple_rnn_10/zeros/packedPackKsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice:output:0Lsequential_21/bidirectional_21/forward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
@sequential_21/bidirectional_21/forward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
:sequential_21/bidirectional_21/forward_simple_rnn_10/zerosFillJsequential_21/bidirectional_21/forward_simple_rnn_10/zeros/packed:output:0Isequential_21/bidirectional_21/forward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Csequential_21/bidirectional_21/forward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ç
>sequential_21/bidirectional_21/forward_simple_rnn_10/transpose	Transposebidirectional_21_inputLsequential_21/bidirectional_21/forward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4®
<sequential_21/bidirectional_21/forward_simple_rnn_10/Shape_1ShapeBsequential_21/bidirectional_21/forward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:
Jsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ä
Dsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_1StridedSliceEsequential_21/bidirectional_21/forward_simple_rnn_10/Shape_1:output:0Ssequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_1/stack:output:0Usequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_1:output:0Usequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Psequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
Bsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2TensorListReserveYsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2/element_shape:output:0Msequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ»
jsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ÿ
\sequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorBsequential_21/bidirectional_21/forward_simple_rnn_10/transpose:y:0ssequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Dsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_2StridedSliceBsequential_21/bidirectional_21/forward_simple_rnn_10/transpose:y:0Ssequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_2/stack:output:0Usequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_1:output:0Usequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
]sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpfsequential_21_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0À
Nsequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMulMatMulMsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_2:output:0esequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
^sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpgsequential_21_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Î
Osequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAddBiasAddXsequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul:product:0fsequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
_sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOphsequential_21_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0º
Psequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1MatMulCsequential_21/bidirectional_21/forward_simple_rnn_10/zeros:output:0gsequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
Ksequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/addAddV2Xsequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd:output:0Zsequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@×
Lsequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/TanhTanhOsequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
Rsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Qsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ä
Dsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1TensorListReserve[sequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0Zsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ{
9sequential_21/bidirectional_21/forward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Msequential_21/bidirectional_21/forward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Gsequential_21/bidirectional_21/forward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
:sequential_21/bidirectional_21/forward_simple_rnn_10/whileWhilePsequential_21/bidirectional_21/forward_simple_rnn_10/while/loop_counter:output:0Vsequential_21/bidirectional_21/forward_simple_rnn_10/while/maximum_iterations:output:0Bsequential_21/bidirectional_21/forward_simple_rnn_10/time:output:0Msequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1:handle:0Csequential_21/bidirectional_21/forward_simple_rnn_10/zeros:output:0Msequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_1:output:0lsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0fsequential_21_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resourcegsequential_21_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resourcehsequential_21_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
Gsequential_21_bidirectional_21_forward_simple_rnn_10_while_body_5007244*S
condKRI
Gsequential_21_bidirectional_21_forward_simple_rnn_10_while_cond_5007243*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¶
esequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   õ
Wsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStackCsequential_21/bidirectional_21/forward_simple_rnn_10/while:output:3nsequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
Jsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Lsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_3StridedSlice`sequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Ssequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_3/stack:output:0Usequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_1:output:0Usequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
Esequential_21/bidirectional_21/forward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
@sequential_21/bidirectional_21/forward_simple_rnn_10/transpose_1	Transpose`sequential_21/bidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Nsequential_21/bidirectional_21/forward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
;sequential_21/bidirectional_21/backward_simple_rnn_10/ShapeShapebidirectional_21_input*
T0*
_output_shapes
:
Isequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ß
Csequential_21/bidirectional_21/backward_simple_rnn_10/strided_sliceStridedSliceDsequential_21/bidirectional_21/backward_simple_rnn_10/Shape:output:0Rsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice/stack:output:0Tsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice/stack_1:output:0Tsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Dsequential_21/bidirectional_21/backward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
Bsequential_21/bidirectional_21/backward_simple_rnn_10/zeros/packedPackLsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice:output:0Msequential_21/bidirectional_21/backward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
Asequential_21/bidirectional_21/backward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
;sequential_21/bidirectional_21/backward_simple_rnn_10/zerosFillKsequential_21/bidirectional_21/backward_simple_rnn_10/zeros/packed:output:0Jsequential_21/bidirectional_21/backward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dsequential_21/bidirectional_21/backward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          é
?sequential_21/bidirectional_21/backward_simple_rnn_10/transpose	Transposebidirectional_21_inputMsequential_21/bidirectional_21/backward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4°
=sequential_21/bidirectional_21/backward_simple_rnn_10/Shape_1ShapeCsequential_21/bidirectional_21/backward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:
Ksequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Msequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Msequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
Esequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_1StridedSliceFsequential_21/bidirectional_21/backward_simple_rnn_10/Shape_1:output:0Tsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_1/stack:output:0Vsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_1:output:0Vsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Qsequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
Csequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2TensorListReserveZsequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2/element_shape:output:0Nsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Dsequential_21/bidirectional_21/backward_simple_rnn_10/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
?sequential_21/bidirectional_21/backward_simple_rnn_10/ReverseV2	ReverseV2Csequential_21/bidirectional_21/backward_simple_rnn_10/transpose:y:0Msequential_21/bidirectional_21/backward_simple_rnn_10/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4¼
ksequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
]sequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorHsequential_21/bidirectional_21/backward_simple_rnn_10/ReverseV2:output:0tsequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Ksequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Msequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Msequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:÷
Esequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_2StridedSliceCsequential_21/bidirectional_21/backward_simple_rnn_10/transpose:y:0Tsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_2/stack:output:0Vsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_1:output:0Vsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
^sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpgsequential_21_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0Ã
Osequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMulMatMulNsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_2:output:0fsequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
_sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOphsequential_21_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ñ
Psequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAddBiasAddYsequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul:product:0gsequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
`sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpisequential_21_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0½
Qsequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1MatMulDsequential_21/bidirectional_21/backward_simple_rnn_10/zeros:output:0hsequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¿
Lsequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/addAddV2Ysequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd:output:0[sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ù
Msequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/TanhTanhPsequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Ssequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Rsequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ç
Esequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1TensorListReserve\sequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0[sequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ|
:sequential_21/bidirectional_21/backward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Nsequential_21/bidirectional_21/backward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Hsequential_21/bidirectional_21/backward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
;sequential_21/bidirectional_21/backward_simple_rnn_10/whileWhileQsequential_21/bidirectional_21/backward_simple_rnn_10/while/loop_counter:output:0Wsequential_21/bidirectional_21/backward_simple_rnn_10/while/maximum_iterations:output:0Csequential_21/bidirectional_21/backward_simple_rnn_10/time:output:0Nsequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1:handle:0Dsequential_21/bidirectional_21/backward_simple_rnn_10/zeros:output:0Nsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_1:output:0msequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0gsequential_21_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resourcehsequential_21_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resourceisequential_21_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *T
bodyLRJ
Hsequential_21_bidirectional_21_backward_simple_rnn_10_while_body_5007352*T
condLRJ
Hsequential_21_bidirectional_21_backward_simple_rnn_10_while_cond_5007351*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ·
fsequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ø
Xsequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStackDsequential_21/bidirectional_21/backward_simple_rnn_10/while:output:3osequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
Ksequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Msequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Msequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Esequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_3StridedSliceasequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Tsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_3/stack:output:0Vsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_1:output:0Vsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
Fsequential_21/bidirectional_21/backward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¸
Asequential_21/bidirectional_21/backward_simple_rnn_10/transpose_1	Transposeasequential_21/bidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Osequential_21/bidirectional_21/backward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
*sequential_21/bidirectional_21/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Á
%sequential_21/bidirectional_21/concatConcatV2Msequential_21/bidirectional_21/forward_simple_rnn_10/strided_slice_3:output:0Nsequential_21/bidirectional_21/backward_simple_rnn_10/strided_slice_3:output:03sequential_21/bidirectional_21/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,sequential_21/dense_21/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_21_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¿
sequential_21/dense_21/MatMulMatMul.sequential_21/bidirectional_21/concat:output:04sequential_21/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_21/dense_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_21/dense_21/BiasAddBiasAdd'sequential_21/dense_21/MatMul:product:05sequential_21/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_21/dense_21/SoftmaxSoftmax'sequential_21/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_21/dense_21/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
NoOpNoOp`^sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp_^sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpa^sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp<^sequential_21/bidirectional_21/backward_simple_rnn_10/while_^sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp^^sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp`^sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp;^sequential_21/bidirectional_21/forward_simple_rnn_10/while.^sequential_21/dense_21/BiasAdd/ReadVariableOp-^sequential_21/dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2Â
_sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp_sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp2À
^sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp^sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp2Ä
`sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp`sequential_21/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp2z
;sequential_21/bidirectional_21/backward_simple_rnn_10/while;sequential_21/bidirectional_21/backward_simple_rnn_10/while2À
^sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp^sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp2¾
]sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp]sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp2Â
_sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp_sequential_21/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp2x
:sequential_21/bidirectional_21/forward_simple_rnn_10/while:sequential_21/bidirectional_21/forward_simple_rnn_10/while2^
-sequential_21/dense_21/BiasAdd/ReadVariableOp-sequential_21/dense_21/BiasAdd/ReadVariableOp2\
,sequential_21/dense_21/MatMul/ReadVariableOp,sequential_21/dense_21/MatMul/ReadVariableOp:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_21_input
ÔB
è
(forward_simple_rnn_10_while_body_5008637H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2G
Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0
forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_10_while_identity*
&forward_simple_rnn_10_while_identity_1*
&forward_simple_rnn_10_while_identity_2*
&forward_simple_rnn_10_while_identity_3*
&forward_simple_rnn_10_while_identity_4E
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp¢Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
Mforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_10_while_placeholderVforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_10/while/simple_rnn_cell_31/MatMulMatMulFforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAddBiasAdd?forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul:product:0Mforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1MatMul)forward_simple_rnn_10_while_placeholder_2Nforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_10/while/simple_rnn_cell_31/addAddV2?forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd:output:0Aforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_10/while/simple_rnn_cell_31/TanhTanh6forward_simple_rnn_10/while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_10_while_placeholder_1Oforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_10/while/addAddV2'forward_simple_rnn_10_while_placeholder*forward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_10/while/add_1AddV2Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counter,forward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_10/while/IdentityIdentity%forward_simple_rnn_10/while/add_1:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_10/while/Identity_1IdentityJforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_10/while/Identity_2Identity#forward_simple_rnn_10/while/add:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_10/while/Identity_3IdentityPforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_10/while/Identity_4Identity7forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0!^forward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_10/while/NoOpNoOpF^forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpE^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpG^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0"U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0"Y
&forward_simple_rnn_10_while_identity_1/forward_simple_rnn_10/while/Identity_1:output:0"Y
&forward_simple_rnn_10_while_identity_2/forward_simple_rnn_10/while/Identity_2:output:0"Y
&forward_simple_rnn_10_while_identity_3/forward_simple_rnn_10/while/Identity_3:output:0"Y
&forward_simple_rnn_10_while_identity_4/forward_simple_rnn_10/while/Identity_4:output:0"¢
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resourcePforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resourceQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resourceOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0"
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpEforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpDforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp2
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpFforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
while_cond_5011323
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5011323___redundant_placeholder05
1while_while_cond_5011323___redundant_placeholder15
1while_while_cond_5011323___redundant_placeholder25
1while_while_cond_5011323___redundant_placeholder3
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
ö©
Ý
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010071
inputs_0Y
Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp¢Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp¢backward_simple_rnn_10/while¢?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp¢@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp¢forward_simple_rnn_10/whileS
forward_simple_rnn_10/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)forward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_10/strided_sliceStridedSlice$forward_simple_rnn_10/Shape:output:02forward_simple_rnn_10/strided_slice/stack:output:04forward_simple_rnn_10/strided_slice/stack_1:output:04forward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_10/zeros/packedPack,forward_simple_rnn_10/strided_slice:output:0-forward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_10/zerosFill+forward_simple_rnn_10/zeros/packed:output:0*forward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
forward_simple_rnn_10/transpose	Transposeinputs_0-forward_simple_rnn_10/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
forward_simple_rnn_10/Shape_1Shape#forward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_10/strided_slice_1StridedSlice&forward_simple_rnn_10/Shape_1:output:04forward_simple_rnn_10/strided_slice_1/stack:output:06forward_simple_rnn_10/strided_slice_1/stack_1:output:06forward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_10/TensorArrayV2TensorListReserve:forward_simple_rnn_10/TensorArrayV2/element_shape:output:0.forward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¢
=forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_10/transpose:y:0Tforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
%forward_simple_rnn_10/strided_slice_2StridedSlice#forward_simple_rnn_10/transpose:y:04forward_simple_rnn_10/strided_slice_2/stack:output:06forward_simple_rnn_10/strided_slice_2/stack_1:output:06forward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_10/simple_rnn_cell_31/MatMulMatMul.forward_simple_rnn_10/strided_slice_2:output:0Fforward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_10/simple_rnn_cell_31/BiasAddBiasAdd9forward_simple_rnn_10/simple_rnn_cell_31/MatMul:product:0Gforward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1MatMul$forward_simple_rnn_10/zeros:output:0Hforward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_10/simple_rnn_cell_31/addAddV29forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd:output:0;forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_10/simple_rnn_cell_31/TanhTanh0forward_simple_rnn_10/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_10/TensorArrayV2_1TensorListReserve<forward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_10/whileWhile1forward_simple_rnn_10/while/loop_counter:output:07forward_simple_rnn_10/while/maximum_iterations:output:0#forward_simple_rnn_10/time:output:0.forward_simple_rnn_10/TensorArrayV2_1:handle:0$forward_simple_rnn_10/zeros:output:0.forward_simple_rnn_10/strided_slice_1:output:0Mforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resourceHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resourceIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_10_while_body_5009894*4
cond,R*
(forward_simple_rnn_10_while_cond_5009893*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_10/while:output:3Oforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_10/strided_slice_3StridedSliceAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_10/strided_slice_3/stack:output:06forward_simple_rnn_10/strided_slice_3/stack_1:output:06forward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_10/transpose_1	TransposeAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
backward_simple_rnn_10/ShapeShapeinputs_0*
T0*
_output_shapes
:t
*backward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_10/strided_sliceStridedSlice%backward_simple_rnn_10/Shape:output:03backward_simple_rnn_10/strided_slice/stack:output:05backward_simple_rnn_10/strided_slice/stack_1:output:05backward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_10/zeros/packedPack-backward_simple_rnn_10/strided_slice:output:0.backward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_10/zerosFill,backward_simple_rnn_10/zeros/packed:output:0+backward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
 backward_simple_rnn_10/transpose	Transposeinputs_0.backward_simple_rnn_10/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr
backward_simple_rnn_10/Shape_1Shape$backward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_10/strided_slice_1StridedSlice'backward_simple_rnn_10/Shape_1:output:05backward_simple_rnn_10/strided_slice_1/stack:output:07backward_simple_rnn_10/strided_slice_1/stack_1:output:07backward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_10/TensorArrayV2TensorListReserve;backward_simple_rnn_10/TensorArrayV2/element_shape:output:0/backward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_10/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ë
 backward_simple_rnn_10/ReverseV2	ReverseV2$backward_simple_rnn_10/transpose:y:0.backward_simple_rnn_10/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Lbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿª
>backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_10/ReverseV2:output:0Ubackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
&backward_simple_rnn_10/strided_slice_2StridedSlice$backward_simple_rnn_10/transpose:y:05backward_simple_rnn_10/strided_slice_2/stack:output:07backward_simple_rnn_10/strided_slice_2/stack_1:output:07backward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÈ
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_10/simple_rnn_cell_32/MatMulMatMul/backward_simple_rnn_10/strided_slice_2:output:0Gbackward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_10/simple_rnn_cell_32/BiasAddBiasAdd:backward_simple_rnn_10/simple_rnn_cell_32/MatMul:product:0Hbackward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1MatMul%backward_simple_rnn_10/zeros:output:0Ibackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_10/simple_rnn_cell_32/addAddV2:backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd:output:0<backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_10/simple_rnn_cell_32/TanhTanh1backward_simple_rnn_10/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_10/TensorArrayV2_1TensorListReserve=backward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_10/whileWhile2backward_simple_rnn_10/while/loop_counter:output:08backward_simple_rnn_10/while/maximum_iterations:output:0$backward_simple_rnn_10/time:output:0/backward_simple_rnn_10/TensorArrayV2_1:handle:0%backward_simple_rnn_10/zeros:output:0/backward_simple_rnn_10/strided_slice_1:output:0Nbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resourceIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resourceJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *5
body-R+
)backward_simple_rnn_10_while_body_5010002*5
cond-R+
)backward_simple_rnn_10_while_cond_5010001*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_10/while:output:3Pbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_10/strided_slice_3StridedSliceBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_10/strided_slice_3/stack:output:07backward_simple_rnn_10/strided_slice_3/stack_1:output:07backward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_10/transpose_1	TransposeBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_10/strided_slice_3:output:0/backward_simple_rnn_10/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@^backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpB^backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp^backward_simple_rnn_10/while@^forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?^forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpA^forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp^forward_simple_rnn_10/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp2
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpAbackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp2<
backward_simple_rnn_10/whilebackward_simple_rnn_10/while2
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp2
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp2:
forward_simple_rnn_10/whileforward_simple_rnn_10/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
þ¨
Û
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5008814

inputsY
Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp¢Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp¢backward_simple_rnn_10/while¢?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp¢@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp¢forward_simple_rnn_10/whileQ
forward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:s
)forward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_10/strided_sliceStridedSlice$forward_simple_rnn_10/Shape:output:02forward_simple_rnn_10/strided_slice/stack:output:04forward_simple_rnn_10/strided_slice/stack_1:output:04forward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_10/zeros/packedPack,forward_simple_rnn_10/strided_slice:output:0-forward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_10/zerosFill+forward_simple_rnn_10/zeros/packed:output:0*forward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_10/transpose	Transposeinputs-forward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
forward_simple_rnn_10/Shape_1Shape#forward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_10/strided_slice_1StridedSlice&forward_simple_rnn_10/Shape_1:output:04forward_simple_rnn_10/strided_slice_1/stack:output:06forward_simple_rnn_10/strided_slice_1/stack_1:output:06forward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_10/TensorArrayV2TensorListReserve:forward_simple_rnn_10/TensorArrayV2/element_shape:output:0.forward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¢
=forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_10/transpose:y:0Tforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%forward_simple_rnn_10/strided_slice_2StridedSlice#forward_simple_rnn_10/transpose:y:04forward_simple_rnn_10/strided_slice_2/stack:output:06forward_simple_rnn_10/strided_slice_2/stack_1:output:06forward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_10/simple_rnn_cell_31/MatMulMatMul.forward_simple_rnn_10/strided_slice_2:output:0Fforward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_10/simple_rnn_cell_31/BiasAddBiasAdd9forward_simple_rnn_10/simple_rnn_cell_31/MatMul:product:0Gforward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1MatMul$forward_simple_rnn_10/zeros:output:0Hforward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_10/simple_rnn_cell_31/addAddV29forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd:output:0;forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_10/simple_rnn_cell_31/TanhTanh0forward_simple_rnn_10/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_10/TensorArrayV2_1TensorListReserve<forward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_10/whileWhile1forward_simple_rnn_10/while/loop_counter:output:07forward_simple_rnn_10/while/maximum_iterations:output:0#forward_simple_rnn_10/time:output:0.forward_simple_rnn_10/TensorArrayV2_1:handle:0$forward_simple_rnn_10/zeros:output:0.forward_simple_rnn_10/strided_slice_1:output:0Mforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resourceHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resourceIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_10_while_body_5008637*4
cond,R*
(forward_simple_rnn_10_while_cond_5008636*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_10/while:output:3Oforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_10/strided_slice_3StridedSliceAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_10/strided_slice_3/stack:output:06forward_simple_rnn_10/strided_slice_3/stack_1:output:06forward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_10/transpose_1	TransposeAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
backward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:t
*backward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_10/strided_sliceStridedSlice%backward_simple_rnn_10/Shape:output:03backward_simple_rnn_10/strided_slice/stack:output:05backward_simple_rnn_10/strided_slice/stack_1:output:05backward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_10/zeros/packedPack-backward_simple_rnn_10/strided_slice:output:0.backward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_10/zerosFill,backward_simple_rnn_10/zeros/packed:output:0+backward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
 backward_simple_rnn_10/transpose	Transposeinputs.backward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4r
backward_simple_rnn_10/Shape_1Shape$backward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_10/strided_slice_1StridedSlice'backward_simple_rnn_10/Shape_1:output:05backward_simple_rnn_10/strided_slice_1/stack:output:07backward_simple_rnn_10/strided_slice_1/stack_1:output:07backward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_10/TensorArrayV2TensorListReserve;backward_simple_rnn_10/TensorArrayV2/element_shape:output:0/backward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_10/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¹
 backward_simple_rnn_10/ReverseV2	ReverseV2$backward_simple_rnn_10/transpose:y:0.backward_simple_rnn_10/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Lbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ª
>backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_10/ReverseV2:output:0Ubackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
&backward_simple_rnn_10/strided_slice_2StridedSlice$backward_simple_rnn_10/transpose:y:05backward_simple_rnn_10/strided_slice_2/stack:output:07backward_simple_rnn_10/strided_slice_2/stack_1:output:07backward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÈ
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_10/simple_rnn_cell_32/MatMulMatMul/backward_simple_rnn_10/strided_slice_2:output:0Gbackward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_10/simple_rnn_cell_32/BiasAddBiasAdd:backward_simple_rnn_10/simple_rnn_cell_32/MatMul:product:0Hbackward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1MatMul%backward_simple_rnn_10/zeros:output:0Ibackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_10/simple_rnn_cell_32/addAddV2:backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd:output:0<backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_10/simple_rnn_cell_32/TanhTanh1backward_simple_rnn_10/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_10/TensorArrayV2_1TensorListReserve=backward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_10/whileWhile2backward_simple_rnn_10/while/loop_counter:output:08backward_simple_rnn_10/while/maximum_iterations:output:0$backward_simple_rnn_10/time:output:0/backward_simple_rnn_10/TensorArrayV2_1:handle:0%backward_simple_rnn_10/zeros:output:0/backward_simple_rnn_10/strided_slice_1:output:0Nbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resourceIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resourceJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *5
body-R+
)backward_simple_rnn_10_while_body_5008745*5
cond-R+
)backward_simple_rnn_10_while_cond_5008744*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_10/while:output:3Pbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_10/strided_slice_3StridedSliceBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_10/strided_slice_3/stack:output:07backward_simple_rnn_10/strided_slice_3/stack_1:output:07backward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_10/transpose_1	TransposeBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_10/strided_slice_3:output:0/backward_simple_rnn_10/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@^backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpB^backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp^backward_simple_rnn_10/while@^forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?^forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpA^forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp^forward_simple_rnn_10/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp2
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpAbackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp2<
backward_simple_rnn_10/whilebackward_simple_rnn_10/while2
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp2
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp2:
forward_simple_rnn_10/whileforward_simple_rnn_10/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
»
Â
8__inference_backward_simple_rnn_10_layer_call_fn_5011279

inputs
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallø
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
GPU2*0J 8 *\
fWRU
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5008410o
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
(forward_simple_rnn_10_while_body_5010114H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2G
Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0
forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_10_while_identity*
&forward_simple_rnn_10_while_identity_1*
&forward_simple_rnn_10_while_identity_2*
&forward_simple_rnn_10_while_identity_3*
&forward_simple_rnn_10_while_identity_4E
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp¢Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
Mforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
?forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_10_while_placeholderVforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ô
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_10/while/simple_rnn_cell_31/MatMulMatMulFforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAddBiasAdd?forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul:product:0Mforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1MatMul)forward_simple_rnn_10_while_placeholder_2Nforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_10/while/simple_rnn_cell_31/addAddV2?forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd:output:0Aforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_10/while/simple_rnn_cell_31/TanhTanh6forward_simple_rnn_10/while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_10_while_placeholder_1Oforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_10/while/addAddV2'forward_simple_rnn_10_while_placeholder*forward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_10/while/add_1AddV2Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counter,forward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_10/while/IdentityIdentity%forward_simple_rnn_10/while/add_1:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_10/while/Identity_1IdentityJforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_10/while/Identity_2Identity#forward_simple_rnn_10/while/add:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_10/while/Identity_3IdentityPforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_10/while/Identity_4Identity7forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0!^forward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_10/while/NoOpNoOpF^forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpE^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpG^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0"U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0"Y
&forward_simple_rnn_10_while_identity_1/forward_simple_rnn_10/while/Identity_1:output:0"Y
&forward_simple_rnn_10_while_identity_2/forward_simple_rnn_10/while/Identity_2:output:0"Y
&forward_simple_rnn_10_while_identity_3/forward_simple_rnn_10/while/Identity_3:output:0"Y
&forward_simple_rnn_10_while_identity_4/forward_simple_rnn_10/while/Identity_4:output:0"¢
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resourcePforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resourceQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resourceOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0"
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpEforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpDforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp2
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpFforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
7__inference_forward_simple_rnn_10_layer_call_fn_5010762
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5007554o
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
Í
ä
)backward_simple_rnn_10_while_cond_5009044J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2L
Hbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5009044___redundant_placeholder0c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5009044___redundant_placeholder1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5009044___redundant_placeholder2c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5009044___redundant_placeholder3)
%backward_simple_rnn_10_while_identity
¾
!backward_simple_rnn_10/while/LessLess(backward_simple_rnn_10_while_placeholderHbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_10/while/IdentityIdentity%backward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0*(
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
4__inference_simple_rnn_cell_32_layer_call_fn_5011803

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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5007774o
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
while_cond_5007787
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5007787___redundant_placeholder05
1while_while_cond_5007787___redundant_placeholder15
1while_while_cond_5007787___redundant_placeholder25
1while_while_cond_5007787___redundant_placeholder3
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
ÏC

)backward_simple_rnn_10_while_body_5010662J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2I
Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_10_while_identity+
'backward_simple_rnn_10_while_identity_1+
'backward_simple_rnn_10_while_identity_2+
'backward_simple_rnn_10_while_identity_3+
'backward_simple_rnn_10_while_identity_4G
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp¢Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
@backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_10_while_placeholderWbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ö
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_10/while/simple_rnn_cell_32/MatMulMatMulGbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAddBiasAdd@backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul:product:0Nbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1MatMul*backward_simple_rnn_10_while_placeholder_2Obackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_10/while/simple_rnn_cell_32/addAddV2@backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd:output:0Bbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_10/while/simple_rnn_cell_32/TanhTanh7backward_simple_rnn_10/while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_10_while_placeholder_1Pbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_10/while/addAddV2(backward_simple_rnn_10_while_placeholder+backward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_10/while/add_1AddV2Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counter-backward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_10/while/IdentityIdentity&backward_simple_rnn_10/while/add_1:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_10/while/Identity_1IdentityLbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_10/while/Identity_2Identity$backward_simple_rnn_10/while/add:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_10/while/Identity_3IdentityQbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_10/while/Identity_4Identity8backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0"^backward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_10/while/NoOpNoOpG^backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpF^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpH^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0"W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0"[
'backward_simple_rnn_10_while_identity_10backward_simple_rnn_10/while/Identity_1:output:0"[
'backward_simple_rnn_10_while_identity_20backward_simple_rnn_10/while/Identity_2:output:0"[
'backward_simple_rnn_10_while_identity_30backward_simple_rnn_10/while/Identity_3:output:0"[
'backward_simple_rnn_10_while_identity_40backward_simple_rnn_10/while/Identity_4:output:0"¤
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resourceQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resourceRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resourcePbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0"
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpFbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpEbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp2
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpGbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
A
Ì
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011727

inputsC
1simple_rnn_cell_32_matmul_readvariableop_resource:4@@
2simple_rnn_cell_32_biasadd_readvariableop_resource:@E
3simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_32/BiasAdd/ReadVariableOp¢(simple_rnn_cell_32/MatMul/ReadVariableOp¢*simple_rnn_cell_32/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_32/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_32/BiasAddBiasAdd#simple_rnn_cell_32/MatMul:product:01simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_32/MatMul_1MatMulzeros:output:02simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_32/addAddV2#simple_rnn_cell_32/BiasAdd:output:0%simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_32/TanhTanhsimple_rnn_cell_32/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_32_matmul_readvariableop_resource2simple_rnn_cell_32_biasadd_readvariableop_resource3simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
while_body_5011660*
condR
while_cond_5011659*8
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
NoOpNoOp*^simple_rnn_cell_32/BiasAdd/ReadVariableOp)^simple_rnn_cell_32/MatMul/ReadVariableOp+^simple_rnn_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_32/BiasAdd/ReadVariableOp)simple_rnn_cell_32/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_32/MatMul/ReadVariableOp(simple_rnn_cell_32/MatMul/ReadVariableOp2X
*simple_rnn_cell_32/MatMul_1/ReadVariableOp*simple_rnn_cell_32/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

Ü
4__inference_simple_rnn_cell_32_layer_call_fn_5011817

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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5007896o
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
þ¨
Û
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5009114

inputsY
Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp¢Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp¢backward_simple_rnn_10/while¢?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp¢@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp¢forward_simple_rnn_10/whileQ
forward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:s
)forward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_10/strided_sliceStridedSlice$forward_simple_rnn_10/Shape:output:02forward_simple_rnn_10/strided_slice/stack:output:04forward_simple_rnn_10/strided_slice/stack_1:output:04forward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_10/zeros/packedPack,forward_simple_rnn_10/strided_slice:output:0-forward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_10/zerosFill+forward_simple_rnn_10/zeros/packed:output:0*forward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_10/transpose	Transposeinputs-forward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
forward_simple_rnn_10/Shape_1Shape#forward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_10/strided_slice_1StridedSlice&forward_simple_rnn_10/Shape_1:output:04forward_simple_rnn_10/strided_slice_1/stack:output:06forward_simple_rnn_10/strided_slice_1/stack_1:output:06forward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_10/TensorArrayV2TensorListReserve:forward_simple_rnn_10/TensorArrayV2/element_shape:output:0.forward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¢
=forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_10/transpose:y:0Tforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%forward_simple_rnn_10/strided_slice_2StridedSlice#forward_simple_rnn_10/transpose:y:04forward_simple_rnn_10/strided_slice_2/stack:output:06forward_simple_rnn_10/strided_slice_2/stack_1:output:06forward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_10/simple_rnn_cell_31/MatMulMatMul.forward_simple_rnn_10/strided_slice_2:output:0Fforward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_10/simple_rnn_cell_31/BiasAddBiasAdd9forward_simple_rnn_10/simple_rnn_cell_31/MatMul:product:0Gforward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1MatMul$forward_simple_rnn_10/zeros:output:0Hforward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_10/simple_rnn_cell_31/addAddV29forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd:output:0;forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_10/simple_rnn_cell_31/TanhTanh0forward_simple_rnn_10/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_10/TensorArrayV2_1TensorListReserve<forward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_10/whileWhile1forward_simple_rnn_10/while/loop_counter:output:07forward_simple_rnn_10/while/maximum_iterations:output:0#forward_simple_rnn_10/time:output:0.forward_simple_rnn_10/TensorArrayV2_1:handle:0$forward_simple_rnn_10/zeros:output:0.forward_simple_rnn_10/strided_slice_1:output:0Mforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resourceHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resourceIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_10_while_body_5008937*4
cond,R*
(forward_simple_rnn_10_while_cond_5008936*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_10/while:output:3Oforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_10/strided_slice_3StridedSliceAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_10/strided_slice_3/stack:output:06forward_simple_rnn_10/strided_slice_3/stack_1:output:06forward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_10/transpose_1	TransposeAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
backward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:t
*backward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_10/strided_sliceStridedSlice%backward_simple_rnn_10/Shape:output:03backward_simple_rnn_10/strided_slice/stack:output:05backward_simple_rnn_10/strided_slice/stack_1:output:05backward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_10/zeros/packedPack-backward_simple_rnn_10/strided_slice:output:0.backward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_10/zerosFill,backward_simple_rnn_10/zeros/packed:output:0+backward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
 backward_simple_rnn_10/transpose	Transposeinputs.backward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4r
backward_simple_rnn_10/Shape_1Shape$backward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_10/strided_slice_1StridedSlice'backward_simple_rnn_10/Shape_1:output:05backward_simple_rnn_10/strided_slice_1/stack:output:07backward_simple_rnn_10/strided_slice_1/stack_1:output:07backward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_10/TensorArrayV2TensorListReserve;backward_simple_rnn_10/TensorArrayV2/element_shape:output:0/backward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_10/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¹
 backward_simple_rnn_10/ReverseV2	ReverseV2$backward_simple_rnn_10/transpose:y:0.backward_simple_rnn_10/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Lbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ª
>backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_10/ReverseV2:output:0Ubackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
&backward_simple_rnn_10/strided_slice_2StridedSlice$backward_simple_rnn_10/transpose:y:05backward_simple_rnn_10/strided_slice_2/stack:output:07backward_simple_rnn_10/strided_slice_2/stack_1:output:07backward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÈ
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_10/simple_rnn_cell_32/MatMulMatMul/backward_simple_rnn_10/strided_slice_2:output:0Gbackward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_10/simple_rnn_cell_32/BiasAddBiasAdd:backward_simple_rnn_10/simple_rnn_cell_32/MatMul:product:0Hbackward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1MatMul%backward_simple_rnn_10/zeros:output:0Ibackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_10/simple_rnn_cell_32/addAddV2:backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd:output:0<backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_10/simple_rnn_cell_32/TanhTanh1backward_simple_rnn_10/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_10/TensorArrayV2_1TensorListReserve=backward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_10/whileWhile2backward_simple_rnn_10/while/loop_counter:output:08backward_simple_rnn_10/while/maximum_iterations:output:0$backward_simple_rnn_10/time:output:0/backward_simple_rnn_10/TensorArrayV2_1:handle:0%backward_simple_rnn_10/zeros:output:0/backward_simple_rnn_10/strided_slice_1:output:0Nbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resourceIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resourceJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *5
body-R+
)backward_simple_rnn_10_while_body_5009045*5
cond-R+
)backward_simple_rnn_10_while_cond_5009044*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_10/while:output:3Pbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_10/strided_slice_3StridedSliceBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_10/strided_slice_3/stack:output:07backward_simple_rnn_10/strided_slice_3/stack_1:output:07backward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_10/transpose_1	TransposeBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_10/strided_slice_3:output:0/backward_simple_rnn_10/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@^backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpB^backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp^backward_simple_rnn_10/while@^forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?^forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpA^forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp^forward_simple_rnn_10/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp2
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpAbackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp2<
backward_simple_rnn_10/whilebackward_simple_rnn_10/while2
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp2
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp2:
forward_simple_rnn_10/whileforward_simple_rnn_10/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ß
¯
while_cond_5011659
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5011659___redundant_placeholder05
1while_while_cond_5011659___redundant_placeholder15
1while_while_cond_5011659___redundant_placeholder25
1while_while_cond_5011659___redundant_placeholder3
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
ØC

)backward_simple_rnn_10_while_body_5010002J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2I
Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_10_while_identity+
'backward_simple_rnn_10_while_identity_1+
'backward_simple_rnn_10_while_identity_2+
'backward_simple_rnn_10_while_identity_3+
'backward_simple_rnn_10_while_identity_4G
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp¢Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ£
@backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_10_while_placeholderWbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ö
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_10/while/simple_rnn_cell_32/MatMulMatMulGbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAddBiasAdd@backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul:product:0Nbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1MatMul*backward_simple_rnn_10_while_placeholder_2Obackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_10/while/simple_rnn_cell_32/addAddV2@backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd:output:0Bbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_10/while/simple_rnn_cell_32/TanhTanh7backward_simple_rnn_10/while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_10_while_placeholder_1Pbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_10/while/addAddV2(backward_simple_rnn_10_while_placeholder+backward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_10/while/add_1AddV2Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counter-backward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_10/while/IdentityIdentity&backward_simple_rnn_10/while/add_1:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_10/while/Identity_1IdentityLbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_10/while/Identity_2Identity$backward_simple_rnn_10/while/add:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_10/while/Identity_3IdentityQbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_10/while/Identity_4Identity8backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0"^backward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_10/while/NoOpNoOpG^backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpF^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpH^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0"W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0"[
'backward_simple_rnn_10_while_identity_10backward_simple_rnn_10/while/Identity_1:output:0"[
'backward_simple_rnn_10_while_identity_20backward_simple_rnn_10/while/Identity_2:output:0"[
'backward_simple_rnn_10_while_identity_30backward_simple_rnn_10/while/Identity_3:output:0"[
'backward_simple_rnn_10_while_identity_40backward_simple_rnn_10/while/Identity_4:output:0"¤
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resourceQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resourceRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resourcePbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0"
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpFbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpEbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp2
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpGbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
while_body_5008475
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_31_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_31/MatMul/ReadVariableOp¢0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_31/BiasAddBiasAdd)while/simple_rnn_cell_31/MatMul:product:07while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_31/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_31/addAddV2)while/simple_rnn_cell_31/BiasAdd:output:0+while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_31/TanhTanh while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_31/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_31/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_31/MatMul/ReadVariableOp1^while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_31_matmul_readvariableop_resource9while_simple_rnn_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_31/MatMul/ReadVariableOp.while/simple_rnn_cell_31/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
7__inference_forward_simple_rnn_10_layer_call_fn_5010784

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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5008140o
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
½"
ß
while_body_5007490
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_31_5007512_0:4@0
"while_simple_rnn_cell_31_5007514_0:@4
"while_simple_rnn_cell_31_5007516_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_31_5007512:4@.
 while_simple_rnn_cell_31_5007514:@2
 while_simple_rnn_cell_31_5007516:@@¢0while/simple_rnn_cell_31/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_31_5007512_0"while_simple_rnn_cell_31_5007514_0"while_simple_rnn_cell_31_5007516_0*
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5007476r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_31/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_31_5007512"while_simple_rnn_cell_31_5007512_0"F
 while_simple_rnn_cell_31_5007514"while_simple_rnn_cell_31_5007514_0"F
 while_simple_rnn_cell_31_5007516"while_simple_rnn_cell_31_5007516_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_31/StatefulPartitionedCall0while/simple_rnn_cell_31/StatefulPartitionedCall: 
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
while_body_5011168
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_31_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_31/MatMul/ReadVariableOp¢0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_31/BiasAddBiasAdd)while/simple_rnn_cell_31/MatMul:product:07while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_31/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_31/addAddV2)while/simple_rnn_cell_31/BiasAdd:output:0+while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_31/TanhTanh while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_31/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_31/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_31/MatMul/ReadVariableOp1^while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_31_matmul_readvariableop_resource9while_simple_rnn_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_31/MatMul/ReadVariableOp.while/simple_rnn_cell_31/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
ÏC

)backward_simple_rnn_10_while_body_5010442J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2I
Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_10_while_identity+
'backward_simple_rnn_10_while_identity_1+
'backward_simple_rnn_10_while_identity_2+
'backward_simple_rnn_10_while_identity_3+
'backward_simple_rnn_10_while_identity_4G
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp¢Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
@backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_10_while_placeholderWbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ö
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_10/while/simple_rnn_cell_32/MatMulMatMulGbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAddBiasAdd@backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul:product:0Nbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1MatMul*backward_simple_rnn_10_while_placeholder_2Obackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_10/while/simple_rnn_cell_32/addAddV2@backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd:output:0Bbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_10/while/simple_rnn_cell_32/TanhTanh7backward_simple_rnn_10/while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_10_while_placeholder_1Pbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_10/while/addAddV2(backward_simple_rnn_10_while_placeholder+backward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_10/while/add_1AddV2Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counter-backward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_10/while/IdentityIdentity&backward_simple_rnn_10/while/add_1:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_10/while/Identity_1IdentityLbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_10/while/Identity_2Identity$backward_simple_rnn_10/while/add:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_10/while/Identity_3IdentityQbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_10/while/Identity_4Identity8backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0"^backward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_10/while/NoOpNoOpG^backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpF^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpH^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0"W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0"[
'backward_simple_rnn_10_while_identity_10backward_simple_rnn_10/while/Identity_1:output:0"[
'backward_simple_rnn_10_while_identity_20backward_simple_rnn_10/while/Identity_2:output:0"[
'backward_simple_rnn_10_while_identity_30backward_simple_rnn_10/while/Identity_3:output:0"[
'backward_simple_rnn_10_while_identity_40backward_simple_rnn_10/while/Identity_4:output:0"¤
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resourceQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resourceRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resourcePbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0"
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpFbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpEbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp2
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpGbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
Í
ä
)backward_simple_rnn_10_while_cond_5010221J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2L
Hbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010221___redundant_placeholder0c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010221___redundant_placeholder1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010221___redundant_placeholder2c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010221___redundant_placeholder3)
%backward_simple_rnn_10_while_identity
¾
!backward_simple_rnn_10/while/LessLess(backward_simple_rnn_10_while_placeholderHbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_10/while/IdentityIdentity%backward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0*(
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
ö©
Ý
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010291
inputs_0Y
Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp¢Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp¢backward_simple_rnn_10/while¢?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp¢@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp¢forward_simple_rnn_10/whileS
forward_simple_rnn_10/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)forward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_10/strided_sliceStridedSlice$forward_simple_rnn_10/Shape:output:02forward_simple_rnn_10/strided_slice/stack:output:04forward_simple_rnn_10/strided_slice/stack_1:output:04forward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_10/zeros/packedPack,forward_simple_rnn_10/strided_slice:output:0-forward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_10/zerosFill+forward_simple_rnn_10/zeros/packed:output:0*forward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
forward_simple_rnn_10/transpose	Transposeinputs_0-forward_simple_rnn_10/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
forward_simple_rnn_10/Shape_1Shape#forward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_10/strided_slice_1StridedSlice&forward_simple_rnn_10/Shape_1:output:04forward_simple_rnn_10/strided_slice_1/stack:output:06forward_simple_rnn_10/strided_slice_1/stack_1:output:06forward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_10/TensorArrayV2TensorListReserve:forward_simple_rnn_10/TensorArrayV2/element_shape:output:0.forward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¢
=forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_10/transpose:y:0Tforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
%forward_simple_rnn_10/strided_slice_2StridedSlice#forward_simple_rnn_10/transpose:y:04forward_simple_rnn_10/strided_slice_2/stack:output:06forward_simple_rnn_10/strided_slice_2/stack_1:output:06forward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_10/simple_rnn_cell_31/MatMulMatMul.forward_simple_rnn_10/strided_slice_2:output:0Fforward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_10/simple_rnn_cell_31/BiasAddBiasAdd9forward_simple_rnn_10/simple_rnn_cell_31/MatMul:product:0Gforward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1MatMul$forward_simple_rnn_10/zeros:output:0Hforward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_10/simple_rnn_cell_31/addAddV29forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd:output:0;forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_10/simple_rnn_cell_31/TanhTanh0forward_simple_rnn_10/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_10/TensorArrayV2_1TensorListReserve<forward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_10/whileWhile1forward_simple_rnn_10/while/loop_counter:output:07forward_simple_rnn_10/while/maximum_iterations:output:0#forward_simple_rnn_10/time:output:0.forward_simple_rnn_10/TensorArrayV2_1:handle:0$forward_simple_rnn_10/zeros:output:0.forward_simple_rnn_10/strided_slice_1:output:0Mforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resourceHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resourceIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_10_while_body_5010114*4
cond,R*
(forward_simple_rnn_10_while_cond_5010113*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_10/while:output:3Oforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_10/strided_slice_3StridedSliceAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_10/strided_slice_3/stack:output:06forward_simple_rnn_10/strided_slice_3/stack_1:output:06forward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_10/transpose_1	TransposeAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
backward_simple_rnn_10/ShapeShapeinputs_0*
T0*
_output_shapes
:t
*backward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_10/strided_sliceStridedSlice%backward_simple_rnn_10/Shape:output:03backward_simple_rnn_10/strided_slice/stack:output:05backward_simple_rnn_10/strided_slice/stack_1:output:05backward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_10/zeros/packedPack-backward_simple_rnn_10/strided_slice:output:0.backward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_10/zerosFill,backward_simple_rnn_10/zeros/packed:output:0+backward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
 backward_simple_rnn_10/transpose	Transposeinputs_0.backward_simple_rnn_10/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr
backward_simple_rnn_10/Shape_1Shape$backward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_10/strided_slice_1StridedSlice'backward_simple_rnn_10/Shape_1:output:05backward_simple_rnn_10/strided_slice_1/stack:output:07backward_simple_rnn_10/strided_slice_1/stack_1:output:07backward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_10/TensorArrayV2TensorListReserve;backward_simple_rnn_10/TensorArrayV2/element_shape:output:0/backward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_10/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ë
 backward_simple_rnn_10/ReverseV2	ReverseV2$backward_simple_rnn_10/transpose:y:0.backward_simple_rnn_10/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Lbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿª
>backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_10/ReverseV2:output:0Ubackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
&backward_simple_rnn_10/strided_slice_2StridedSlice$backward_simple_rnn_10/transpose:y:05backward_simple_rnn_10/strided_slice_2/stack:output:07backward_simple_rnn_10/strided_slice_2/stack_1:output:07backward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÈ
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_10/simple_rnn_cell_32/MatMulMatMul/backward_simple_rnn_10/strided_slice_2:output:0Gbackward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_10/simple_rnn_cell_32/BiasAddBiasAdd:backward_simple_rnn_10/simple_rnn_cell_32/MatMul:product:0Hbackward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1MatMul%backward_simple_rnn_10/zeros:output:0Ibackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_10/simple_rnn_cell_32/addAddV2:backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd:output:0<backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_10/simple_rnn_cell_32/TanhTanh1backward_simple_rnn_10/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_10/TensorArrayV2_1TensorListReserve=backward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_10/whileWhile2backward_simple_rnn_10/while/loop_counter:output:08backward_simple_rnn_10/while/maximum_iterations:output:0$backward_simple_rnn_10/time:output:0/backward_simple_rnn_10/TensorArrayV2_1:handle:0%backward_simple_rnn_10/zeros:output:0/backward_simple_rnn_10/strided_slice_1:output:0Nbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resourceIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resourceJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *5
body-R+
)backward_simple_rnn_10_while_body_5010222*5
cond-R+
)backward_simple_rnn_10_while_cond_5010221*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_10/while:output:3Pbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_10/strided_slice_3StridedSliceBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_10/strided_slice_3/stack:output:07backward_simple_rnn_10/strided_slice_3/stack_1:output:07backward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_10/transpose_1	TransposeBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_10/strided_slice_3:output:0/backward_simple_rnn_10/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@^backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpB^backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp^backward_simple_rnn_10/while@^forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?^forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpA^forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp^forward_simple_rnn_10/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp2
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpAbackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp2<
backward_simple_rnn_10/whilebackward_simple_rnn_10/while2
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp2
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp2:
forward_simple_rnn_10/whileforward_simple_rnn_10/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
?
Ë
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011125

inputsC
1simple_rnn_cell_31_matmul_readvariableop_resource:4@@
2simple_rnn_cell_31_biasadd_readvariableop_resource:@E
3simple_rnn_cell_31_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_31/BiasAdd/ReadVariableOp¢(simple_rnn_cell_31/MatMul/ReadVariableOp¢*simple_rnn_cell_31/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_31/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_31/BiasAddBiasAdd#simple_rnn_cell_31/MatMul:product:01simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_31/MatMul_1MatMulzeros:output:02simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_31/addAddV2#simple_rnn_cell_31/BiasAdd:output:0%simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_31/TanhTanhsimple_rnn_cell_31/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_31_matmul_readvariableop_resource2simple_rnn_cell_31_biasadd_readvariableop_resource3simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
while_body_5011058*
condR
while_cond_5011057*8
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
NoOpNoOp*^simple_rnn_cell_31/BiasAdd/ReadVariableOp)^simple_rnn_cell_31/MatMul/ReadVariableOp+^simple_rnn_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_31/BiasAdd/ReadVariableOp)simple_rnn_cell_31/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_31/MatMul/ReadVariableOp(simple_rnn_cell_31/MatMul/ReadVariableOp2X
*simple_rnn_cell_31/MatMul_1/ReadVariableOp*simple_rnn_cell_31/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
Ñ
(forward_simple_rnn_10_while_cond_5009893H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2J
Fforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5009893___redundant_placeholder0a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5009893___redundant_placeholder1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5009893___redundant_placeholder2a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5009893___redundant_placeholder3(
$forward_simple_rnn_10_while_identity
º
 forward_simple_rnn_10/while/LessLess'forward_simple_rnn_10_while_placeholderFforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_10/while/IdentityIdentity$forward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0*(
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
·	

2__inference_bidirectional_21_layer_call_fn_5009800
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5008270p
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
ß
¯
while_cond_5008342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5008342___redundant_placeholder05
1while_while_cond_5008342___redundant_placeholder15
1while_while_cond_5008342___redundant_placeholder25
1while_while_cond_5008342___redundant_placeholder3
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
ÿ6
­
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5008015

inputs,
simple_rnn_cell_32_5007938:4@(
simple_rnn_cell_32_5007940:@,
simple_rnn_cell_32_5007942:@@
identity¢*simple_rnn_cell_32/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_32_5007938simple_rnn_cell_32_5007940simple_rnn_cell_32_5007942*
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5007896n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_32_5007938simple_rnn_cell_32_5007940simple_rnn_cell_32_5007942*
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
while_body_5007951*
condR
while_cond_5007950*8
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
NoOpNoOp+^simple_rnn_cell_32/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_32/StatefulPartitionedCall*simple_rnn_cell_32/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ÿ6
­
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5007852

inputs,
simple_rnn_cell_32_5007775:4@(
simple_rnn_cell_32_5007777:@,
simple_rnn_cell_32_5007779:@@
identity¢*simple_rnn_cell_32/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_32_5007775simple_rnn_cell_32_5007777simple_rnn_cell_32_5007779*
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5007774n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_32_5007775simple_rnn_cell_32_5007777simple_rnn_cell_32_5007779*
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
while_body_5007788*
condR
while_cond_5007787*8
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
NoOpNoOp+^simple_rnn_cell_32/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_32/StatefulPartitionedCall*simple_rnn_cell_32/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
û>
Í
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011015
inputs_0C
1simple_rnn_cell_31_matmul_readvariableop_resource:4@@
2simple_rnn_cell_31_biasadd_readvariableop_resource:@E
3simple_rnn_cell_31_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_31/BiasAdd/ReadVariableOp¢(simple_rnn_cell_31/MatMul/ReadVariableOp¢*simple_rnn_cell_31/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_31/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_31/BiasAddBiasAdd#simple_rnn_cell_31/MatMul:product:01simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_31/MatMul_1MatMulzeros:output:02simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_31/addAddV2#simple_rnn_cell_31/BiasAdd:output:0%simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_31/TanhTanhsimple_rnn_cell_31/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_31_matmul_readvariableop_resource2simple_rnn_cell_31_biasadd_readvariableop_resource3simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
while_body_5010948*
condR
while_cond_5010947*8
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
NoOpNoOp*^simple_rnn_cell_31/BiasAdd/ReadVariableOp)^simple_rnn_cell_31/MatMul/ReadVariableOp+^simple_rnn_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_31/BiasAdd/ReadVariableOp)simple_rnn_cell_31/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_31/MatMul/ReadVariableOp(simple_rnn_cell_31/MatMul/ReadVariableOp2X
*simple_rnn_cell_31/MatMul_1/ReadVariableOp*simple_rnn_cell_31/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
Í
ä
)backward_simple_rnn_10_while_cond_5010001J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2L
Hbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010001___redundant_placeholder0c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010001___redundant_placeholder1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010001___redundant_placeholder2c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010001___redundant_placeholder3)
%backward_simple_rnn_10_while_identity
¾
!backward_simple_rnn_10/while/LessLess(backward_simple_rnn_10_while_placeholderHbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_10/while/IdentityIdentity%backward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0*(
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
¢Ó
ý
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009556

inputsj
Xbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource:4@g
Ybidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource:@l
Zbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@k
Ybidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource:4@h
Zbidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource:@m
[bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@:
'dense_21_matmul_readvariableop_resource:	6
(dense_21_biasadd_readvariableop_resource:
identity¢Qbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Pbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp¢Rbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp¢-bidirectional_21/backward_simple_rnn_10/while¢Pbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Obidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp¢Qbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp¢,bidirectional_21/forward_simple_rnn_10/while¢dense_21/BiasAdd/ReadVariableOp¢dense_21/MatMul/ReadVariableOpb
,bidirectional_21/forward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:
:bidirectional_21/forward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<bidirectional_21/forward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<bidirectional_21/forward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4bidirectional_21/forward_simple_rnn_10/strided_sliceStridedSlice5bidirectional_21/forward_simple_rnn_10/Shape:output:0Cbidirectional_21/forward_simple_rnn_10/strided_slice/stack:output:0Ebidirectional_21/forward_simple_rnn_10/strided_slice/stack_1:output:0Ebidirectional_21/forward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5bidirectional_21/forward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@è
3bidirectional_21/forward_simple_rnn_10/zeros/packedPack=bidirectional_21/forward_simple_rnn_10/strided_slice:output:0>bidirectional_21/forward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:w
2bidirectional_21/forward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    á
,bidirectional_21/forward_simple_rnn_10/zerosFill<bidirectional_21/forward_simple_rnn_10/zeros/packed:output:0;bidirectional_21/forward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5bidirectional_21/forward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
0bidirectional_21/forward_simple_rnn_10/transpose	Transposeinputs>bidirectional_21/forward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
.bidirectional_21/forward_simple_rnn_10/Shape_1Shape4bidirectional_21/forward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:
<bidirectional_21/forward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_21/forward_simple_rnn_10/strided_slice_1StridedSlice7bidirectional_21/forward_simple_rnn_10/Shape_1:output:0Ebidirectional_21/forward_simple_rnn_10/strided_slice_1/stack:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_1:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bbidirectional_21/forward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
4bidirectional_21/forward_simple_rnn_10/TensorArrayV2TensorListReserveKbidirectional_21/forward_simple_rnn_10/TensorArrayV2/element_shape:output:0?bidirectional_21/forward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ­
\bidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Õ
Nbidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor4bidirectional_21/forward_simple_rnn_10/transpose:y:0ebidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<bidirectional_21/forward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
6bidirectional_21/forward_simple_rnn_10/strided_slice_2StridedSlice4bidirectional_21/forward_simple_rnn_10/transpose:y:0Ebidirectional_21/forward_simple_rnn_10/strided_slice_2/stack:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_1:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskè
Obidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpXbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
@bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMulMatMul?bidirectional_21/forward_simple_rnn_10/strided_slice_2:output:0Wbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Pbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpYbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Abidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAddBiasAddJbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul:product:0Xbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ì
Qbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpZbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Bbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1MatMul5bidirectional_21/forward_simple_rnn_10/zeros:output:0Ybidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
=bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/addAddV2Jbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd:output:0Lbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
>bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/TanhTanhAbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dbidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Cbidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :º
6bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1TensorListReserveMbidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0Lbidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
+bidirectional_21/forward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 
?bidirectional_21/forward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ{
9bidirectional_21/forward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø	
,bidirectional_21/forward_simple_rnn_10/whileWhileBbidirectional_21/forward_simple_rnn_10/while/loop_counter:output:0Hbidirectional_21/forward_simple_rnn_10/while/maximum_iterations:output:04bidirectional_21/forward_simple_rnn_10/time:output:0?bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1:handle:05bidirectional_21/forward_simple_rnn_10/zeros:output:0?bidirectional_21/forward_simple_rnn_10/strided_slice_1:output:0^bidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Xbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resourceYbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resourceZbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
9bidirectional_21_forward_simple_rnn_10_while_body_5009372*E
cond=R;
9bidirectional_21_forward_simple_rnn_10_while_cond_5009371*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¨
Wbidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
Ibidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack5bidirectional_21/forward_simple_rnn_10/while:output:3`bidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
<bidirectional_21/forward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
>bidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ê
6bidirectional_21/forward_simple_rnn_10/strided_slice_3StridedSliceRbidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Ebidirectional_21/forward_simple_rnn_10/strided_slice_3/stack:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_1:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
7bidirectional_21/forward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
2bidirectional_21/forward_simple_rnn_10/transpose_1	TransposeRbidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional_21/forward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
-bidirectional_21/backward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:
;bidirectional_21/backward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_21/backward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_21/backward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_21/backward_simple_rnn_10/strided_sliceStridedSlice6bidirectional_21/backward_simple_rnn_10/Shape:output:0Dbidirectional_21/backward_simple_rnn_10/strided_slice/stack:output:0Fbidirectional_21/backward_simple_rnn_10/strided_slice/stack_1:output:0Fbidirectional_21/backward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6bidirectional_21/backward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@ë
4bidirectional_21/backward_simple_rnn_10/zeros/packedPack>bidirectional_21/backward_simple_rnn_10/strided_slice:output:0?bidirectional_21/backward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:x
3bidirectional_21/backward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ä
-bidirectional_21/backward_simple_rnn_10/zerosFill=bidirectional_21/backward_simple_rnn_10/zeros/packed:output:0<bidirectional_21/backward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
6bidirectional_21/backward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
1bidirectional_21/backward_simple_rnn_10/transpose	Transposeinputs?bidirectional_21/backward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
/bidirectional_21/backward_simple_rnn_10/Shape_1Shape5bidirectional_21/backward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:
=bidirectional_21/backward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?bidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7bidirectional_21/backward_simple_rnn_10/strided_slice_1StridedSlice8bidirectional_21/backward_simple_rnn_10/Shape_1:output:0Fbidirectional_21/backward_simple_rnn_10/strided_slice_1/stack:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_1:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Cbidirectional_21/backward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¬
5bidirectional_21/backward_simple_rnn_10/TensorArrayV2TensorListReserveLbidirectional_21/backward_simple_rnn_10/TensorArrayV2/element_shape:output:0@bidirectional_21/backward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
6bidirectional_21/backward_simple_rnn_10/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ì
1bidirectional_21/backward_simple_rnn_10/ReverseV2	ReverseV25bidirectional_21/backward_simple_rnn_10/transpose:y:0?bidirectional_21/backward_simple_rnn_10/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4®
]bidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ý
Obidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor:bidirectional_21/backward_simple_rnn_10/ReverseV2:output:0fbidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=bidirectional_21/backward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?bidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:±
7bidirectional_21/backward_simple_rnn_10/strided_slice_2StridedSlice5bidirectional_21/backward_simple_rnn_10/transpose:y:0Fbidirectional_21/backward_simple_rnn_10/strided_slice_2/stack:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_1:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskê
Pbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpYbidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
Abidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMulMatMul@bidirectional_21/backward_simple_rnn_10/strided_slice_2:output:0Xbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@è
Qbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpZbidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
Bbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAddBiasAddKbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul:product:0Ybidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
Rbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp[bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Cbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1MatMul6bidirectional_21/backward_simple_rnn_10/zeros:output:0Zbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/addAddV2Kbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd:output:0Mbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@½
?bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/TanhTanhBbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Ebidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Dbidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :½
7bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1TensorListReserveNbidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0Mbidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
,bidirectional_21/backward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 
@bidirectional_21/backward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ|
:bidirectional_21/backward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : å	
-bidirectional_21/backward_simple_rnn_10/whileWhileCbidirectional_21/backward_simple_rnn_10/while/loop_counter:output:0Ibidirectional_21/backward_simple_rnn_10/while/maximum_iterations:output:05bidirectional_21/backward_simple_rnn_10/time:output:0@bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1:handle:06bidirectional_21/backward_simple_rnn_10/zeros:output:0@bidirectional_21/backward_simple_rnn_10/strided_slice_1:output:0_bidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ybidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resourceZbidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource[bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *F
body>R<
:bidirectional_21_backward_simple_rnn_10_while_body_5009480*F
cond>R<
:bidirectional_21_backward_simple_rnn_10_while_cond_5009479*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ©
Xbidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Î
Jbidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack6bidirectional_21/backward_simple_rnn_10/while:output:3abidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
=bidirectional_21/backward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
?bidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
7bidirectional_21/backward_simple_rnn_10/strided_slice_3StridedSliceSbidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Fbidirectional_21/backward_simple_rnn_10/strided_slice_3/stack:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_1:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
8bidirectional_21/backward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
3bidirectional_21/backward_simple_rnn_10/transpose_1	TransposeSbidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Abidirectional_21/backward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
bidirectional_21/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
bidirectional_21/concatConcatV2?bidirectional_21/forward_simple_rnn_10/strided_slice_3:output:0@bidirectional_21/backward_simple_rnn_10/strided_slice_3:output:0%bidirectional_21/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_21/MatMulMatMul bidirectional_21/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_21/SoftmaxSoftmaxdense_21/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_21/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOpR^bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpQ^bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpS^bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp.^bidirectional_21/backward_simple_rnn_10/whileQ^bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpP^bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpR^bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp-^bidirectional_21/forward_simple_rnn_10/while ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2¦
Qbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpQbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp2¤
Pbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpPbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp2¨
Rbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpRbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp2^
-bidirectional_21/backward_simple_rnn_10/while-bidirectional_21/backward_simple_rnn_10/while2¤
Pbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpPbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp2¢
Obidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpObidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp2¦
Qbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpQbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp2\
,bidirectional_21/forward_simple_rnn_10/while,bidirectional_21/forward_simple_rnn_10/while2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
½"
ß
while_body_5007788
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_32_5007810_0:4@0
"while_simple_rnn_cell_32_5007812_0:@4
"while_simple_rnn_cell_32_5007814_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_32_5007810:4@.
 while_simple_rnn_cell_32_5007812:@2
 while_simple_rnn_cell_32_5007814:@@¢0while/simple_rnn_cell_32/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_32/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_32_5007810_0"while_simple_rnn_cell_32_5007812_0"while_simple_rnn_cell_32_5007814_0*
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5007774r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_32/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_32/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_32_5007810"while_simple_rnn_cell_32_5007810_0"F
 while_simple_rnn_cell_32_5007812"while_simple_rnn_cell_32_5007812_0"F
 while_simple_rnn_cell_32_5007814"while_simple_rnn_cell_32_5007814_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_32/StatefulPartitionedCall0while/simple_rnn_cell_32/StatefulPartitionedCall: 
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
9bidirectional_21_forward_simple_rnn_10_while_cond_5009371j
fbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_loop_counterp
lbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_maximum_iterations<
8bidirectional_21_forward_simple_rnn_10_while_placeholder>
:bidirectional_21_forward_simple_rnn_10_while_placeholder_1>
:bidirectional_21_forward_simple_rnn_10_while_placeholder_2l
hbidirectional_21_forward_simple_rnn_10_while_less_bidirectional_21_forward_simple_rnn_10_strided_slice_1
bidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_cond_5009371___redundant_placeholder0
bidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_cond_5009371___redundant_placeholder1
bidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_cond_5009371___redundant_placeholder2
bidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_cond_5009371___redundant_placeholder39
5bidirectional_21_forward_simple_rnn_10_while_identity
þ
1bidirectional_21/forward_simple_rnn_10/while/LessLess8bidirectional_21_forward_simple_rnn_10_while_placeholderhbidirectional_21_forward_simple_rnn_10_while_less_bidirectional_21_forward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: 
5bidirectional_21/forward_simple_rnn_10/while/IdentityIdentity5bidirectional_21/forward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "w
5bidirectional_21_forward_simple_rnn_10_while_identity>bidirectional_21/forward_simple_rnn_10/while/Identity:output:0*(
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
(forward_simple_rnn_10_while_body_5008937H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2G
Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0
forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_10_while_identity*
&forward_simple_rnn_10_while_identity_1*
&forward_simple_rnn_10_while_identity_2*
&forward_simple_rnn_10_while_identity_3*
&forward_simple_rnn_10_while_identity_4E
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp¢Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
Mforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_10_while_placeholderVforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_10/while/simple_rnn_cell_31/MatMulMatMulFforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAddBiasAdd?forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul:product:0Mforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1MatMul)forward_simple_rnn_10_while_placeholder_2Nforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_10/while/simple_rnn_cell_31/addAddV2?forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd:output:0Aforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_10/while/simple_rnn_cell_31/TanhTanh6forward_simple_rnn_10/while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_10_while_placeholder_1Oforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_10/while/addAddV2'forward_simple_rnn_10_while_placeholder*forward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_10/while/add_1AddV2Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counter,forward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_10/while/IdentityIdentity%forward_simple_rnn_10/while/add_1:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_10/while/Identity_1IdentityJforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_10/while/Identity_2Identity#forward_simple_rnn_10/while/add:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_10/while/Identity_3IdentityPforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_10/while/Identity_4Identity7forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0!^forward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_10/while/NoOpNoOpF^forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpE^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpG^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0"U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0"Y
&forward_simple_rnn_10_while_identity_1/forward_simple_rnn_10/while/Identity_1:output:0"Y
&forward_simple_rnn_10_while_identity_2/forward_simple_rnn_10/while/Identity_2:output:0"Y
&forward_simple_rnn_10_while_identity_3/forward_simple_rnn_10/while/Identity_3:output:0"Y
&forward_simple_rnn_10_while_identity_4/forward_simple_rnn_10/while/Identity_4:output:0"¢
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resourcePforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resourceQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resourceOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0"
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpEforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpDforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp2
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpFforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
ä

J__inference_sequential_21_layer_call_and_return_conditional_losses_5009236
bidirectional_21_input*
bidirectional_21_5009217:4@&
bidirectional_21_5009219:@*
bidirectional_21_5009221:@@*
bidirectional_21_5009223:4@&
bidirectional_21_5009225:@*
bidirectional_21_5009227:@@#
dense_21_5009230:	
dense_21_5009232:
identity¢(bidirectional_21/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall
(bidirectional_21/StatefulPartitionedCallStatefulPartitionedCallbidirectional_21_inputbidirectional_21_5009217bidirectional_21_5009219bidirectional_21_5009221bidirectional_21_5009223bidirectional_21_5009225bidirectional_21_5009227*
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5008814¡
 dense_21/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_21/StatefulPartitionedCall:output:0dense_21_5009230dense_21_5009232*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_5008839x
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_21/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_21/StatefulPartitionedCall(bidirectional_21/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_21_input
Í
ä
)backward_simple_rnn_10_while_cond_5010661J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2L
Hbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010661___redundant_placeholder0c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010661___redundant_placeholder1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010661___redundant_placeholder2c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5010661___redundant_placeholder3)
%backward_simple_rnn_10_while_identity
¾
!backward_simple_rnn_10/while/LessLess(backward_simple_rnn_10_while_placeholderHbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_10/while/IdentityIdentity%backward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0*(
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
4__inference_simple_rnn_cell_31_layer_call_fn_5011755

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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5007598o
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
ü-
Ò
while_body_5008343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_32_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_32/MatMul/ReadVariableOp¢0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_32/BiasAddBiasAdd)while/simple_rnn_cell_32/MatMul:product:07while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_32/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_32/addAddV2)while/simple_rnn_cell_32/BiasAdd:output:0+while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_32/TanhTanh while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_32/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_32/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_32/MatMul/ReadVariableOp1^while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_32_matmul_readvariableop_resource9while_simple_rnn_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_32/MatMul/ReadVariableOp.while/simple_rnn_cell_32/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
?
Ë
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5008542

inputsC
1simple_rnn_cell_31_matmul_readvariableop_resource:4@@
2simple_rnn_cell_31_biasadd_readvariableop_resource:@E
3simple_rnn_cell_31_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_31/BiasAdd/ReadVariableOp¢(simple_rnn_cell_31/MatMul/ReadVariableOp¢*simple_rnn_cell_31/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_31/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_31/BiasAddBiasAdd#simple_rnn_cell_31/MatMul:product:01simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_31/MatMul_1MatMulzeros:output:02simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_31/addAddV2#simple_rnn_cell_31/BiasAdd:output:0%simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_31/TanhTanhsimple_rnn_cell_31/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_31_matmul_readvariableop_resource2simple_rnn_cell_31_biasadd_readvariableop_resource3simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
while_body_5008475*
condR
while_cond_5008474*8
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
NoOpNoOp*^simple_rnn_cell_31/BiasAdd/ReadVariableOp)^simple_rnn_cell_31/MatMul/ReadVariableOp+^simple_rnn_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_31/BiasAdd/ReadVariableOp)simple_rnn_cell_31/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_31/MatMul/ReadVariableOp(simple_rnn_cell_31/MatMul/ReadVariableOp2X
*simple_rnn_cell_31/MatMul_1/ReadVariableOp*simple_rnn_cell_31/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü-
Ò
while_body_5008073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_31_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_31/MatMul/ReadVariableOp¢0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_31/BiasAddBiasAdd)while/simple_rnn_cell_31/MatMul:product:07while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_31/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_31/addAddV2)while/simple_rnn_cell_31/BiasAdd:output:0+while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_31/TanhTanh while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_31/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_31/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_31/MatMul/ReadVariableOp1^while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_31_biasadd_readvariableop_resource:while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_31_matmul_1_readvariableop_resource;while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_31_matmul_readvariableop_resource9while_simple_rnn_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_31/MatMul/ReadVariableOp.while/simple_rnn_cell_31/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp0while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
(forward_simple_rnn_10_while_cond_5008936H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2J
Fforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5008936___redundant_placeholder0a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5008936___redundant_placeholder1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5008936___redundant_placeholder2a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5008936___redundant_placeholder3(
$forward_simple_rnn_10_while_identity
º
 forward_simple_rnn_10/while/LessLess'forward_simple_rnn_10_while_placeholderFforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_10/while/IdentityIdentity$forward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0*(
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
while_cond_5011435
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5011435___redundant_placeholder05
1while_while_cond_5011435___redundant_placeholder15
1while_while_cond_5011435___redundant_placeholder25
1while_while_cond_5011435___redundant_placeholder3
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
S
þ
 __inference__traced_save_5011973
file_prefix.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop_
[savev2_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_read_readvariableopi
esavev2_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_read_readvariableop]
Ysavev2_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_read_readvariableop`
\savev2_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_read_readvariableopj
fsavev2_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_read_readvariableop^
Zsavev2_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableopf
bsavev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_m_read_readvariableopp
lsavev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_m_read_readvariableopd
`savev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_m_read_readvariableopg
csavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_m_read_readvariableopq
msavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_m_read_readvariableope
asavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableopf
bsavev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_v_read_readvariableopp
lsavev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_v_read_readvariableopd
`savev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_v_read_readvariableopg
csavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_v_read_readvariableopq
msavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_v_read_readvariableope
asavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_v_read_readvariableop
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
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Þ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop[savev2_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_read_readvariableopesavev2_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_read_readvariableopYsavev2_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_read_readvariableop\savev2_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_read_readvariableopfsavev2_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_read_readvariableopZsavev2_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableopbsavev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_m_read_readvariableoplsavev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_m_read_readvariableop`savev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_m_read_readvariableopcsavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_m_read_readvariableopmsavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_m_read_readvariableopasavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableopbsavev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_kernel_v_read_readvariableoplsavev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_recurrent_kernel_v_read_readvariableop`savev2_adam_bidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_bias_v_read_readvariableopcsavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_kernel_v_read_readvariableopmsavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_recurrent_kernel_v_read_readvariableopasavev2_adam_bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
ï`
µ
Hsequential_21_bidirectional_21_backward_simple_rnn_10_while_body_5007352
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_loop_counter
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_maximum_iterationsK
Gsequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholderM
Isequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholder_1M
Isequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholder_2
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_strided_slice_1_0Ä
¿sequential_21_bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0
osequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@~
psequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@
qsequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@H
Dsequential_21_bidirectional_21_backward_simple_rnn_10_while_identityJ
Fsequential_21_bidirectional_21_backward_simple_rnn_10_while_identity_1J
Fsequential_21_bidirectional_21_backward_simple_rnn_10_while_identity_2J
Fsequential_21_bidirectional_21_backward_simple_rnn_10_while_identity_3J
Fsequential_21_bidirectional_21_backward_simple_rnn_10_while_identity_4
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_strided_slice_1Â
½sequential_21_bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor
msequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource:4@|
nsequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource:@
osequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢esequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢dsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp¢fsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp¾
msequential_21/bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   µ
_sequential_21/bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¿sequential_21_bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0Gsequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholdervsequential_21/bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0
dsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOposequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0ç
Usequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMulMatMulfsequential_21/bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0lsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
esequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOppsequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0ã
Vsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAddBiasAdd_sequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul:product:0msequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
fsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpqsequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Î
Wsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1MatMulIsequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholder_2nsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ñ
Rsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/addAddV2_sequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd:output:0asequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@å
Ssequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/TanhTanhVsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
fsequential_21/bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
`sequential_21/bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemIsequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholder_1osequential_21/bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:0Wsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ
Asequential_21/bidirectional_21/backward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :þ
?sequential_21/bidirectional_21/backward_simple_rnn_10/while/addAddV2Gsequential_21_bidirectional_21_backward_simple_rnn_10_while_placeholderJsequential_21/bidirectional_21/backward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: 
Csequential_21/bidirectional_21/backward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :À
Asequential_21/bidirectional_21/backward_simple_rnn_10/while/add_1AddV2sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_loop_counterLsequential_21/bidirectional_21/backward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: û
Dsequential_21/bidirectional_21/backward_simple_rnn_10/while/IdentityIdentityEsequential_21/bidirectional_21/backward_simple_rnn_10/while/add_1:z:0A^sequential_21/bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ã
Fsequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity_1Identitysequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_while_maximum_iterationsA^sequential_21/bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: û
Fsequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity_2IdentityCsequential_21/bidirectional_21/backward_simple_rnn_10/while/add:z:0A^sequential_21/bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ¨
Fsequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity_3Identitypsequential_21/bidirectional_21/backward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0A^sequential_21/bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
:  
Fsequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity_4IdentityWsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0A^sequential_21/bidirectional_21/backward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
@sequential_21/bidirectional_21/backward_simple_rnn_10/while/NoOpNoOpf^sequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpe^sequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpg^sequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Dsequential_21_bidirectional_21_backward_simple_rnn_10_while_identityMsequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity:output:0"
Fsequential_21_bidirectional_21_backward_simple_rnn_10_while_identity_1Osequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity_1:output:0"
Fsequential_21_bidirectional_21_backward_simple_rnn_10_while_identity_2Osequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity_2:output:0"
Fsequential_21_bidirectional_21_backward_simple_rnn_10_while_identity_3Osequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity_3:output:0"
Fsequential_21_bidirectional_21_backward_simple_rnn_10_while_identity_4Osequential_21/bidirectional_21/backward_simple_rnn_10/while/Identity_4:output:0"
sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_strided_slice_1sequential_21_bidirectional_21_backward_simple_rnn_10_while_sequential_21_bidirectional_21_backward_simple_rnn_10_strided_slice_1_0"â
nsequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resourcepsequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"ä
osequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resourceqsequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"à
msequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resourceosequential_21_bidirectional_21_backward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0"
½sequential_21_bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor¿sequential_21_bidirectional_21_backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2Î
esequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpesequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2Ì
dsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpdsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp2Ð
fsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpfsequential_21/bidirectional_21/backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
Gsequential_21_bidirectional_21_forward_simple_rnn_10_while_body_5007244
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_loop_counter
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_maximum_iterationsJ
Fsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholderL
Hsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholder_1L
Hsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholder_2
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_strided_slice_1_0Â
½sequential_21_bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0
nsequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@}
osequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@
psequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@G
Csequential_21_bidirectional_21_forward_simple_rnn_10_while_identityI
Esequential_21_bidirectional_21_forward_simple_rnn_10_while_identity_1I
Esequential_21_bidirectional_21_forward_simple_rnn_10_while_identity_2I
Esequential_21_bidirectional_21_forward_simple_rnn_10_while_identity_3I
Esequential_21_bidirectional_21_forward_simple_rnn_10_while_identity_4
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_strided_slice_1À
»sequential_21_bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor~
lsequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource:4@{
msequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource:@
nsequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢dsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢csequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp¢esequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp½
lsequential_21/bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   °
^sequential_21/bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem½sequential_21_bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0Fsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholderusequential_21/bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0
csequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpnsequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0ä
Tsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMulMatMulesequential_21/bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0ksequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOposequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0à
Usequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAddBiasAdd^sequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul:product:0lsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
esequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOppsequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Ë
Vsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1MatMulHsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholder_2msequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
Qsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/addAddV2^sequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd:output:0`sequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ã
Rsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/TanhTanhUsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
esequential_21/bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
_sequential_21/bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemHsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholder_1nsequential_21/bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:0Vsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ
@sequential_21/bidirectional_21/forward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :û
>sequential_21/bidirectional_21/forward_simple_rnn_10/while/addAddV2Fsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholderIsequential_21/bidirectional_21/forward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: 
Bsequential_21/bidirectional_21/forward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¼
@sequential_21/bidirectional_21/forward_simple_rnn_10/while/add_1AddV2sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_loop_counterKsequential_21/bidirectional_21/forward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: ø
Csequential_21/bidirectional_21/forward_simple_rnn_10/while/IdentityIdentityDsequential_21/bidirectional_21/forward_simple_rnn_10/while/add_1:z:0@^sequential_21/bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ¿
Esequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity_1Identitysequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_maximum_iterations@^sequential_21/bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ø
Esequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity_2IdentityBsequential_21/bidirectional_21/forward_simple_rnn_10/while/add:z:0@^sequential_21/bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ¥
Esequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity_3Identityosequential_21/bidirectional_21/forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0@^sequential_21/bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
Esequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity_4IdentityVsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0@^sequential_21/bidirectional_21/forward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
?sequential_21/bidirectional_21/forward_simple_rnn_10/while/NoOpNoOpe^sequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpd^sequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpf^sequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Csequential_21_bidirectional_21_forward_simple_rnn_10_while_identityLsequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity:output:0"
Esequential_21_bidirectional_21_forward_simple_rnn_10_while_identity_1Nsequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity_1:output:0"
Esequential_21_bidirectional_21_forward_simple_rnn_10_while_identity_2Nsequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity_2:output:0"
Esequential_21_bidirectional_21_forward_simple_rnn_10_while_identity_3Nsequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity_3:output:0"
Esequential_21_bidirectional_21_forward_simple_rnn_10_while_identity_4Nsequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity_4:output:0"
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_strided_slice_1sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_strided_slice_1_0"à
msequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resourceosequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"â
nsequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resourcepsequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0"Þ
lsequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resourcensequential_21_bidirectional_21_forward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0"þ
»sequential_21_bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor½sequential_21_bidirectional_21_forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_21_bidirectional_21_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2Ì
dsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpdsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2Ê
csequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpcsequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp2Î
esequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpesequential_21/bidirectional_21/forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
(forward_simple_rnn_10_while_cond_5010333H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2J
Fforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010333___redundant_placeholder0a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010333___redundant_placeholder1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010333___redundant_placeholder2a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5010333___redundant_placeholder3(
$forward_simple_rnn_10_while_identity
º
 forward_simple_rnn_10/while/LessLess'forward_simple_rnn_10_while_placeholderFforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_10/while/IdentityIdentity$forward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0*(
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
A
Ì
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5008410

inputsC
1simple_rnn_cell_32_matmul_readvariableop_resource:4@@
2simple_rnn_cell_32_biasadd_readvariableop_resource:@E
3simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_32/BiasAdd/ReadVariableOp¢(simple_rnn_cell_32/MatMul/ReadVariableOp¢*simple_rnn_cell_32/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_32/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_32/BiasAddBiasAdd#simple_rnn_cell_32/MatMul:product:01simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_32/MatMul_1MatMulzeros:output:02simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_32/addAddV2#simple_rnn_cell_32/BiasAdd:output:0%simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_32/TanhTanhsimple_rnn_cell_32/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_32_matmul_readvariableop_resource2simple_rnn_cell_32_biasadd_readvariableop_resource3simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
while_body_5008343*
condR
while_cond_5008342*8
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
NoOpNoOp*^simple_rnn_cell_32/BiasAdd/ReadVariableOp)^simple_rnn_cell_32/MatMul/ReadVariableOp+^simple_rnn_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_32/BiasAdd/ReadVariableOp)simple_rnn_cell_32/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_32/MatMul/ReadVariableOp(simple_rnn_cell_32/MatMul/ReadVariableOp2X
*simple_rnn_cell_32/MatMul_1/ReadVariableOp*simple_rnn_cell_32/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÝB
è
(forward_simple_rnn_10_while_body_5009894H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2G
Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0
forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_10_while_identity*
&forward_simple_rnn_10_while_identity_1*
&forward_simple_rnn_10_while_identity_2*
&forward_simple_rnn_10_while_identity_3*
&forward_simple_rnn_10_while_identity_4E
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp¢Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp
Mforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
?forward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_10_while_placeholderVforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ô
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_10/while/simple_rnn_cell_31/MatMulMatMulFforward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAddBiasAdd?forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul:product:0Mforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1MatMul)forward_simple_rnn_10_while_placeholder_2Nforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_10/while/simple_rnn_cell_31/addAddV2?forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd:output:0Aforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_10/while/simple_rnn_cell_31/TanhTanh6forward_simple_rnn_10/while/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_10_while_placeholder_1Oforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_10/while/addAddV2'forward_simple_rnn_10_while_placeholder*forward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_10/while/add_1AddV2Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counter,forward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_10/while/IdentityIdentity%forward_simple_rnn_10/while/add_1:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_10/while/Identity_1IdentityJforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_10/while/Identity_2Identity#forward_simple_rnn_10/while/add:z:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_10/while/Identity_3IdentityPforward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_10/while/Identity_4Identity7forward_simple_rnn_10/while/simple_rnn_cell_31/Tanh:y:0!^forward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_10/while/NoOpNoOpF^forward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpE^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpG^forward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1Cforward_simple_rnn_10_while_forward_simple_rnn_10_strided_slice_1_0"U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0"Y
&forward_simple_rnn_10_while_identity_1/forward_simple_rnn_10/while/Identity_1:output:0"Y
&forward_simple_rnn_10_while_identity_2/forward_simple_rnn_10/while/Identity_2:output:0"Y
&forward_simple_rnn_10_while_identity_3/forward_simple_rnn_10/while/Identity_3:output:0"Y
&forward_simple_rnn_10_while_identity_4/forward_simple_rnn_10/while/Identity_4:output:0"¢
Nforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resourcePforward_simple_rnn_10_while_simple_rnn_cell_31_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resourceQforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resourceOforward_simple_rnn_10_while_simple_rnn_cell_31_matmul_readvariableop_resource_0"
}forward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOpEforward_simple_rnn_10/while/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOpDforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul/ReadVariableOp2
Fforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOpFforward_simple_rnn_10/while/simple_rnn_cell_31/MatMul_1/ReadVariableOp: 
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
Ú@
Î
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011503
inputs_0C
1simple_rnn_cell_32_matmul_readvariableop_resource:4@@
2simple_rnn_cell_32_biasadd_readvariableop_resource:@E
3simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_32/BiasAdd/ReadVariableOp¢(simple_rnn_cell_32/MatMul/ReadVariableOp¢*simple_rnn_cell_32/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_32/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_32/BiasAddBiasAdd#simple_rnn_cell_32/MatMul:product:01simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_32/MatMul_1MatMulzeros:output:02simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_32/addAddV2#simple_rnn_cell_32/BiasAdd:output:0%simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_32/TanhTanhsimple_rnn_cell_32/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_32_matmul_readvariableop_resource2simple_rnn_cell_32_biasadd_readvariableop_resource3simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
while_body_5011436*
condR
while_cond_5011435*8
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
NoOpNoOp*^simple_rnn_cell_32/BiasAdd/ReadVariableOp)^simple_rnn_cell_32/MatMul/ReadVariableOp+^simple_rnn_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_32/BiasAdd/ReadVariableOp)simple_rnn_cell_32/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_32/MatMul/ReadVariableOp(simple_rnn_cell_32/MatMul/ReadVariableOp2X
*simple_rnn_cell_32/MatMul_1/ReadVariableOp*simple_rnn_cell_32/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
þ¨
Û
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010511

inputsY
Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp¢Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp¢backward_simple_rnn_10/while¢?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp¢@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp¢forward_simple_rnn_10/whileQ
forward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:s
)forward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_10/strided_sliceStridedSlice$forward_simple_rnn_10/Shape:output:02forward_simple_rnn_10/strided_slice/stack:output:04forward_simple_rnn_10/strided_slice/stack_1:output:04forward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_10/zeros/packedPack,forward_simple_rnn_10/strided_slice:output:0-forward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_10/zerosFill+forward_simple_rnn_10/zeros/packed:output:0*forward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_10/transpose	Transposeinputs-forward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
forward_simple_rnn_10/Shape_1Shape#forward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_10/strided_slice_1StridedSlice&forward_simple_rnn_10/Shape_1:output:04forward_simple_rnn_10/strided_slice_1/stack:output:06forward_simple_rnn_10/strided_slice_1/stack_1:output:06forward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_10/TensorArrayV2TensorListReserve:forward_simple_rnn_10/TensorArrayV2/element_shape:output:0.forward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¢
=forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_10/transpose:y:0Tforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%forward_simple_rnn_10/strided_slice_2StridedSlice#forward_simple_rnn_10/transpose:y:04forward_simple_rnn_10/strided_slice_2/stack:output:06forward_simple_rnn_10/strided_slice_2/stack_1:output:06forward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_10/simple_rnn_cell_31/MatMulMatMul.forward_simple_rnn_10/strided_slice_2:output:0Fforward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_10/simple_rnn_cell_31/BiasAddBiasAdd9forward_simple_rnn_10/simple_rnn_cell_31/MatMul:product:0Gforward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1MatMul$forward_simple_rnn_10/zeros:output:0Hforward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_10/simple_rnn_cell_31/addAddV29forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd:output:0;forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_10/simple_rnn_cell_31/TanhTanh0forward_simple_rnn_10/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_10/TensorArrayV2_1TensorListReserve<forward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_10/whileWhile1forward_simple_rnn_10/while/loop_counter:output:07forward_simple_rnn_10/while/maximum_iterations:output:0#forward_simple_rnn_10/time:output:0.forward_simple_rnn_10/TensorArrayV2_1:handle:0$forward_simple_rnn_10/zeros:output:0.forward_simple_rnn_10/strided_slice_1:output:0Mforward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resourceHforward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resourceIforward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_10_while_body_5010334*4
cond,R*
(forward_simple_rnn_10_while_cond_5010333*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_10/while:output:3Oforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_10/strided_slice_3StridedSliceAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_10/strided_slice_3/stack:output:06forward_simple_rnn_10/strided_slice_3/stack_1:output:06forward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_10/transpose_1	TransposeAforward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
backward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:t
*backward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_10/strided_sliceStridedSlice%backward_simple_rnn_10/Shape:output:03backward_simple_rnn_10/strided_slice/stack:output:05backward_simple_rnn_10/strided_slice/stack_1:output:05backward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_10/zeros/packedPack-backward_simple_rnn_10/strided_slice:output:0.backward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_10/zerosFill,backward_simple_rnn_10/zeros/packed:output:0+backward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
 backward_simple_rnn_10/transpose	Transposeinputs.backward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4r
backward_simple_rnn_10/Shape_1Shape$backward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_10/strided_slice_1StridedSlice'backward_simple_rnn_10/Shape_1:output:05backward_simple_rnn_10/strided_slice_1/stack:output:07backward_simple_rnn_10/strided_slice_1/stack_1:output:07backward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_10/TensorArrayV2TensorListReserve;backward_simple_rnn_10/TensorArrayV2/element_shape:output:0/backward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_10/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¹
 backward_simple_rnn_10/ReverseV2	ReverseV2$backward_simple_rnn_10/transpose:y:0.backward_simple_rnn_10/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Lbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ª
>backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_10/ReverseV2:output:0Ubackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
&backward_simple_rnn_10/strided_slice_2StridedSlice$backward_simple_rnn_10/transpose:y:05backward_simple_rnn_10/strided_slice_2/stack:output:07backward_simple_rnn_10/strided_slice_2/stack_1:output:07backward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÈ
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_10/simple_rnn_cell_32/MatMulMatMul/backward_simple_rnn_10/strided_slice_2:output:0Gbackward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_10/simple_rnn_cell_32/BiasAddBiasAdd:backward_simple_rnn_10/simple_rnn_cell_32/MatMul:product:0Hbackward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1MatMul%backward_simple_rnn_10/zeros:output:0Ibackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_10/simple_rnn_cell_32/addAddV2:backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd:output:0<backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_10/simple_rnn_cell_32/TanhTanh1backward_simple_rnn_10/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_10/TensorArrayV2_1TensorListReserve=backward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_10/whileWhile2backward_simple_rnn_10/while/loop_counter:output:08backward_simple_rnn_10/while/maximum_iterations:output:0$backward_simple_rnn_10/time:output:0/backward_simple_rnn_10/TensorArrayV2_1:handle:0%backward_simple_rnn_10/zeros:output:0/backward_simple_rnn_10/strided_slice_1:output:0Nbackward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resourceIbackward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resourceJbackward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *5
body-R+
)backward_simple_rnn_10_while_body_5010442*5
cond-R+
)backward_simple_rnn_10_while_cond_5010441*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_10/while:output:3Pbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_10/strided_slice_3StridedSliceBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_10/strided_slice_3/stack:output:07backward_simple_rnn_10/strided_slice_3/stack_1:output:07backward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_10/transpose_1	TransposeBbackward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_10/strided_slice_3:output:0/backward_simple_rnn_10/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@^backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpB^backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp^backward_simple_rnn_10/while@^forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?^forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpA^forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp^forward_simple_rnn_10/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp@backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp?backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp2
Abackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpAbackward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp2<
backward_simple_rnn_10/whilebackward_simple_rnn_10/while2
?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp?forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp2
>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp>forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp2
@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp@forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp2:
forward_simple_rnn_10/whileforward_simple_rnn_10/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
¢Ó
ý
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009783

inputsj
Xbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource:4@g
Ybidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource:@l
Zbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource:@@k
Ybidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource:4@h
Zbidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource:@m
[bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@:
'dense_21_matmul_readvariableop_resource:	6
(dense_21_biasadd_readvariableop_resource:
identity¢Qbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Pbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp¢Rbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp¢-bidirectional_21/backward_simple_rnn_10/while¢Pbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp¢Obidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp¢Qbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp¢,bidirectional_21/forward_simple_rnn_10/while¢dense_21/BiasAdd/ReadVariableOp¢dense_21/MatMul/ReadVariableOpb
,bidirectional_21/forward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:
:bidirectional_21/forward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<bidirectional_21/forward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<bidirectional_21/forward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4bidirectional_21/forward_simple_rnn_10/strided_sliceStridedSlice5bidirectional_21/forward_simple_rnn_10/Shape:output:0Cbidirectional_21/forward_simple_rnn_10/strided_slice/stack:output:0Ebidirectional_21/forward_simple_rnn_10/strided_slice/stack_1:output:0Ebidirectional_21/forward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5bidirectional_21/forward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@è
3bidirectional_21/forward_simple_rnn_10/zeros/packedPack=bidirectional_21/forward_simple_rnn_10/strided_slice:output:0>bidirectional_21/forward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:w
2bidirectional_21/forward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    á
,bidirectional_21/forward_simple_rnn_10/zerosFill<bidirectional_21/forward_simple_rnn_10/zeros/packed:output:0;bidirectional_21/forward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5bidirectional_21/forward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
0bidirectional_21/forward_simple_rnn_10/transpose	Transposeinputs>bidirectional_21/forward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
.bidirectional_21/forward_simple_rnn_10/Shape_1Shape4bidirectional_21/forward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:
<bidirectional_21/forward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_21/forward_simple_rnn_10/strided_slice_1StridedSlice7bidirectional_21/forward_simple_rnn_10/Shape_1:output:0Ebidirectional_21/forward_simple_rnn_10/strided_slice_1/stack:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_1:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bbidirectional_21/forward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
4bidirectional_21/forward_simple_rnn_10/TensorArrayV2TensorListReserveKbidirectional_21/forward_simple_rnn_10/TensorArrayV2/element_shape:output:0?bidirectional_21/forward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ­
\bidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Õ
Nbidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor4bidirectional_21/forward_simple_rnn_10/transpose:y:0ebidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<bidirectional_21/forward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
6bidirectional_21/forward_simple_rnn_10/strided_slice_2StridedSlice4bidirectional_21/forward_simple_rnn_10/transpose:y:0Ebidirectional_21/forward_simple_rnn_10/strided_slice_2/stack:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_1:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskè
Obidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpReadVariableOpXbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
@bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMulMatMul?bidirectional_21/forward_simple_rnn_10/strided_slice_2:output:0Wbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Pbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpReadVariableOpYbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Abidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAddBiasAddJbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul:product:0Xbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ì
Qbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpReadVariableOpZbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Bbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1MatMul5bidirectional_21/forward_simple_rnn_10/zeros:output:0Ybidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
=bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/addAddV2Jbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd:output:0Lbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
>bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/TanhTanhAbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dbidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Cbidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :º
6bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1TensorListReserveMbidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0Lbidirectional_21/forward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
+bidirectional_21/forward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 
?bidirectional_21/forward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ{
9bidirectional_21/forward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø	
,bidirectional_21/forward_simple_rnn_10/whileWhileBbidirectional_21/forward_simple_rnn_10/while/loop_counter:output:0Hbidirectional_21/forward_simple_rnn_10/while/maximum_iterations:output:04bidirectional_21/forward_simple_rnn_10/time:output:0?bidirectional_21/forward_simple_rnn_10/TensorArrayV2_1:handle:05bidirectional_21/forward_simple_rnn_10/zeros:output:0?bidirectional_21/forward_simple_rnn_10/strided_slice_1:output:0^bidirectional_21/forward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Xbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_readvariableop_resourceYbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_biasadd_readvariableop_resourceZbidirectional_21_forward_simple_rnn_10_simple_rnn_cell_31_matmul_1_readvariableop_resource*
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
9bidirectional_21_forward_simple_rnn_10_while_body_5009599*E
cond=R;
9bidirectional_21_forward_simple_rnn_10_while_cond_5009598*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¨
Wbidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
Ibidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack5bidirectional_21/forward_simple_rnn_10/while:output:3`bidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
<bidirectional_21/forward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
>bidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ê
6bidirectional_21/forward_simple_rnn_10/strided_slice_3StridedSliceRbidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Ebidirectional_21/forward_simple_rnn_10/strided_slice_3/stack:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_1:output:0Gbidirectional_21/forward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
7bidirectional_21/forward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
2bidirectional_21/forward_simple_rnn_10/transpose_1	TransposeRbidirectional_21/forward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional_21/forward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
-bidirectional_21/backward_simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:
;bidirectional_21/backward_simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_21/backward_simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_21/backward_simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_21/backward_simple_rnn_10/strided_sliceStridedSlice6bidirectional_21/backward_simple_rnn_10/Shape:output:0Dbidirectional_21/backward_simple_rnn_10/strided_slice/stack:output:0Fbidirectional_21/backward_simple_rnn_10/strided_slice/stack_1:output:0Fbidirectional_21/backward_simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6bidirectional_21/backward_simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@ë
4bidirectional_21/backward_simple_rnn_10/zeros/packedPack>bidirectional_21/backward_simple_rnn_10/strided_slice:output:0?bidirectional_21/backward_simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:x
3bidirectional_21/backward_simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ä
-bidirectional_21/backward_simple_rnn_10/zerosFill=bidirectional_21/backward_simple_rnn_10/zeros/packed:output:0<bidirectional_21/backward_simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
6bidirectional_21/backward_simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
1bidirectional_21/backward_simple_rnn_10/transpose	Transposeinputs?bidirectional_21/backward_simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
/bidirectional_21/backward_simple_rnn_10/Shape_1Shape5bidirectional_21/backward_simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:
=bidirectional_21/backward_simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?bidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7bidirectional_21/backward_simple_rnn_10/strided_slice_1StridedSlice8bidirectional_21/backward_simple_rnn_10/Shape_1:output:0Fbidirectional_21/backward_simple_rnn_10/strided_slice_1/stack:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_1:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Cbidirectional_21/backward_simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¬
5bidirectional_21/backward_simple_rnn_10/TensorArrayV2TensorListReserveLbidirectional_21/backward_simple_rnn_10/TensorArrayV2/element_shape:output:0@bidirectional_21/backward_simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
6bidirectional_21/backward_simple_rnn_10/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ì
1bidirectional_21/backward_simple_rnn_10/ReverseV2	ReverseV25bidirectional_21/backward_simple_rnn_10/transpose:y:0?bidirectional_21/backward_simple_rnn_10/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4®
]bidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ý
Obidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor:bidirectional_21/backward_simple_rnn_10/ReverseV2:output:0fbidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=bidirectional_21/backward_simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?bidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:±
7bidirectional_21/backward_simple_rnn_10/strided_slice_2StridedSlice5bidirectional_21/backward_simple_rnn_10/transpose:y:0Fbidirectional_21/backward_simple_rnn_10/strided_slice_2/stack:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_1:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskê
Pbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpYbidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
Abidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMulMatMul@bidirectional_21/backward_simple_rnn_10/strided_slice_2:output:0Xbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@è
Qbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpZbidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
Bbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAddBiasAddKbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul:product:0Ybidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
Rbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp[bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Cbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1MatMul6bidirectional_21/backward_simple_rnn_10/zeros:output:0Zbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/addAddV2Kbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd:output:0Mbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@½
?bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/TanhTanhBbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Ebidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Dbidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :½
7bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1TensorListReserveNbidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/element_shape:output:0Mbidirectional_21/backward_simple_rnn_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
,bidirectional_21/backward_simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 
@bidirectional_21/backward_simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ|
:bidirectional_21/backward_simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : å	
-bidirectional_21/backward_simple_rnn_10/whileWhileCbidirectional_21/backward_simple_rnn_10/while/loop_counter:output:0Ibidirectional_21/backward_simple_rnn_10/while/maximum_iterations:output:05bidirectional_21/backward_simple_rnn_10/time:output:0@bidirectional_21/backward_simple_rnn_10/TensorArrayV2_1:handle:06bidirectional_21/backward_simple_rnn_10/zeros:output:0@bidirectional_21/backward_simple_rnn_10/strided_slice_1:output:0_bidirectional_21/backward_simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ybidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_readvariableop_resourceZbidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_biasadd_readvariableop_resource[bidirectional_21_backward_simple_rnn_10_simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *F
body>R<
:bidirectional_21_backward_simple_rnn_10_while_body_5009707*F
cond>R<
:bidirectional_21_backward_simple_rnn_10_while_cond_5009706*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ©
Xbidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Î
Jbidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack6bidirectional_21/backward_simple_rnn_10/while:output:3abidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
=bidirectional_21/backward_simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
?bidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
7bidirectional_21/backward_simple_rnn_10/strided_slice_3StridedSliceSbidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Fbidirectional_21/backward_simple_rnn_10/strided_slice_3/stack:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_1:output:0Hbidirectional_21/backward_simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
8bidirectional_21/backward_simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
3bidirectional_21/backward_simple_rnn_10/transpose_1	TransposeSbidirectional_21/backward_simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0Abidirectional_21/backward_simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
bidirectional_21/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
bidirectional_21/concatConcatV2?bidirectional_21/forward_simple_rnn_10/strided_slice_3:output:0@bidirectional_21/backward_simple_rnn_10/strided_slice_3:output:0%bidirectional_21/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_21/MatMulMatMul bidirectional_21/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_21/SoftmaxSoftmaxdense_21/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_21/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOpR^bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpQ^bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpS^bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp.^bidirectional_21/backward_simple_rnn_10/whileQ^bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpP^bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpR^bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp-^bidirectional_21/forward_simple_rnn_10/while ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2¦
Qbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOpQbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/BiasAdd/ReadVariableOp2¤
Pbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOpPbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul/ReadVariableOp2¨
Rbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOpRbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/MatMul_1/ReadVariableOp2^
-bidirectional_21/backward_simple_rnn_10/while-bidirectional_21/backward_simple_rnn_10/while2¤
Pbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOpPbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/BiasAdd/ReadVariableOp2¢
Obidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOpObidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul/ReadVariableOp2¦
Qbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOpQbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/MatMul_1/ReadVariableOp2\
,bidirectional_21/forward_simple_rnn_10/while,bidirectional_21/forward_simple_rnn_10/while2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs

ì
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5011772

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
E__inference_dense_21_layer_call_and_return_conditional_losses_5008839

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
A
Ì
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011615

inputsC
1simple_rnn_cell_32_matmul_readvariableop_resource:4@@
2simple_rnn_cell_32_biasadd_readvariableop_resource:@E
3simple_rnn_cell_32_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_32/BiasAdd/ReadVariableOp¢(simple_rnn_cell_32/MatMul/ReadVariableOp¢*simple_rnn_cell_32/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_32_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_32/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_32/BiasAddBiasAdd#simple_rnn_cell_32/MatMul:product:01simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_32_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_32/MatMul_1MatMulzeros:output:02simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_32/addAddV2#simple_rnn_cell_32/BiasAdd:output:0%simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_32/TanhTanhsimple_rnn_cell_32/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_32_matmul_readvariableop_resource2simple_rnn_cell_32_biasadd_readvariableop_resource3simple_rnn_cell_32_matmul_1_readvariableop_resource*
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
while_body_5011548*
condR
while_cond_5011547*8
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
NoOpNoOp*^simple_rnn_cell_32/BiasAdd/ReadVariableOp)^simple_rnn_cell_32/MatMul/ReadVariableOp+^simple_rnn_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_32/BiasAdd/ReadVariableOp)simple_rnn_cell_32/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_32/MatMul/ReadVariableOp(simple_rnn_cell_32/MatMul/ReadVariableOp2X
*simple_rnn_cell_32/MatMul_1/ReadVariableOp*simple_rnn_cell_32/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
¯
while_cond_5011167
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5011167___redundant_placeholder05
1while_while_cond_5011167___redundant_placeholder15
1while_while_cond_5011167___redundant_placeholder25
1while_while_cond_5011167___redundant_placeholder3
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
 5
¬
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5007554

inputs,
simple_rnn_cell_31_5007477:4@(
simple_rnn_cell_31_5007479:@,
simple_rnn_cell_31_5007481:@@
identity¢*simple_rnn_cell_31/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_31_5007477simple_rnn_cell_31_5007479simple_rnn_cell_31_5007481*
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5007476n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_31_5007477simple_rnn_cell_31_5007479simple_rnn_cell_31_5007481*
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
while_body_5007490*
condR
while_cond_5007489*8
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
NoOpNoOp+^simple_rnn_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_31/StatefulPartitionedCall*simple_rnn_cell_31/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs

	
9bidirectional_21_forward_simple_rnn_10_while_cond_5009598j
fbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_loop_counterp
lbidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_maximum_iterations<
8bidirectional_21_forward_simple_rnn_10_while_placeholder>
:bidirectional_21_forward_simple_rnn_10_while_placeholder_1>
:bidirectional_21_forward_simple_rnn_10_while_placeholder_2l
hbidirectional_21_forward_simple_rnn_10_while_less_bidirectional_21_forward_simple_rnn_10_strided_slice_1
bidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_cond_5009598___redundant_placeholder0
bidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_cond_5009598___redundant_placeholder1
bidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_cond_5009598___redundant_placeholder2
bidirectional_21_forward_simple_rnn_10_while_bidirectional_21_forward_simple_rnn_10_while_cond_5009598___redundant_placeholder39
5bidirectional_21_forward_simple_rnn_10_while_identity
þ
1bidirectional_21/forward_simple_rnn_10/while/LessLess8bidirectional_21_forward_simple_rnn_10_while_placeholderhbidirectional_21_forward_simple_rnn_10_while_less_bidirectional_21_forward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: 
5bidirectional_21/forward_simple_rnn_10/while/IdentityIdentity5bidirectional_21/forward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "w
5bidirectional_21_forward_simple_rnn_10_while_identity>bidirectional_21/forward_simple_rnn_10/while/Identity:output:0*(
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
while_cond_5008474
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5008474___redundant_placeholder05
1while_while_cond_5008474___redundant_placeholder15
1while_while_cond_5008474___redundant_placeholder25
1while_while_cond_5008474___redundant_placeholder3
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

¬
Gsequential_21_bidirectional_21_forward_simple_rnn_10_while_cond_5007243
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_loop_counter
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_maximum_iterationsJ
Fsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholderL
Hsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholder_1L
Hsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholder_2
sequential_21_bidirectional_21_forward_simple_rnn_10_while_less_sequential_21_bidirectional_21_forward_simple_rnn_10_strided_slice_1 
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_cond_5007243___redundant_placeholder0 
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_cond_5007243___redundant_placeholder1 
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_cond_5007243___redundant_placeholder2 
sequential_21_bidirectional_21_forward_simple_rnn_10_while_sequential_21_bidirectional_21_forward_simple_rnn_10_while_cond_5007243___redundant_placeholder3G
Csequential_21_bidirectional_21_forward_simple_rnn_10_while_identity
·
?sequential_21/bidirectional_21/forward_simple_rnn_10/while/LessLessFsequential_21_bidirectional_21_forward_simple_rnn_10_while_placeholdersequential_21_bidirectional_21_forward_simple_rnn_10_while_less_sequential_21_bidirectional_21_forward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: µ
Csequential_21/bidirectional_21/forward_simple_rnn_10/while/IdentityIdentityCsequential_21/bidirectional_21/forward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "
Csequential_21_bidirectional_21_forward_simple_rnn_10_while_identityLsequential_21/bidirectional_21/forward_simple_rnn_10/while/Identity:output:0*(
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
(forward_simple_rnn_10_while_cond_5008636H
Dforward_simple_rnn_10_while_forward_simple_rnn_10_while_loop_counterN
Jforward_simple_rnn_10_while_forward_simple_rnn_10_while_maximum_iterations+
'forward_simple_rnn_10_while_placeholder-
)forward_simple_rnn_10_while_placeholder_1-
)forward_simple_rnn_10_while_placeholder_2J
Fforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5008636___redundant_placeholder0a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5008636___redundant_placeholder1a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5008636___redundant_placeholder2a
]forward_simple_rnn_10_while_forward_simple_rnn_10_while_cond_5008636___redundant_placeholder3(
$forward_simple_rnn_10_while_identity
º
 forward_simple_rnn_10/while/LessLess'forward_simple_rnn_10_while_placeholderFforward_simple_rnn_10_while_less_forward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_10/while/IdentityIdentity$forward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_10_while_identity-forward_simple_rnn_10/while/Identity:output:0*(
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
Í
ä
)backward_simple_rnn_10_while_cond_5008744J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2L
Hbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5008744___redundant_placeholder0c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5008744___redundant_placeholder1c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5008744___redundant_placeholder2c
_backward_simple_rnn_10_while_backward_simple_rnn_10_while_cond_5008744___redundant_placeholder3)
%backward_simple_rnn_10_while_identity
¾
!backward_simple_rnn_10/while/LessLess(backward_simple_rnn_10_while_placeholderHbackward_simple_rnn_10_while_less_backward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_10/while/IdentityIdentity%backward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0*(
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
while_cond_5007489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5007489___redundant_placeholder05
1while_while_cond_5007489___redundant_placeholder15
1while_while_cond_5007489___redundant_placeholder25
1while_while_cond_5007489___redundant_placeholder3
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
ü-
Ò
while_body_5011660
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_32_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_32/MatMul/ReadVariableOp¢0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_32/BiasAddBiasAdd)while/simple_rnn_cell_32/MatMul:product:07while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_32/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_32/addAddV2)while/simple_rnn_cell_32/BiasAdd:output:0+while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_32/TanhTanh while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_32/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_32/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_32/MatMul/ReadVariableOp1^while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_32_matmul_readvariableop_resource9while_simple_rnn_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_32/MatMul/ReadVariableOp.while/simple_rnn_cell_32/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
while_body_5011548
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_32_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_32/MatMul/ReadVariableOp¢0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_32/BiasAddBiasAdd)while/simple_rnn_cell_32/MatMul:product:07while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_32/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_32/addAddV2)while/simple_rnn_cell_32/BiasAdd:output:0+while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_32/TanhTanh while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_32/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_32/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_32/MatMul/ReadVariableOp1^while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_32_matmul_readvariableop_resource9while_simple_rnn_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_32/MatMul/ReadVariableOp.while/simple_rnn_cell_32/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
¯
Ä
8__inference_backward_simple_rnn_10_layer_call_fn_5011246
inputs_0
unknown:4@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallú
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
GPU2*0J 8 *\
fWRU
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5007852o
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

î
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5008573

inputs/
forward_simple_rnn_10_5008556:4@+
forward_simple_rnn_10_5008558:@/
forward_simple_rnn_10_5008560:@@0
backward_simple_rnn_10_5008563:4@,
backward_simple_rnn_10_5008565:@0
backward_simple_rnn_10_5008567:@@
identity¢.backward_simple_rnn_10/StatefulPartitionedCall¢-forward_simple_rnn_10/StatefulPartitionedCallË
-forward_simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_10_5008556forward_simple_rnn_10_5008558forward_simple_rnn_10_5008560*
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5008542Ð
.backward_simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_10_5008563backward_simple_rnn_10_5008565backward_simple_rnn_10_5008567*
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
GPU2*0J 8 *\
fWRU
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5008410M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Õ
concatConcatV26forward_simple_rnn_10/StatefulPartitionedCall:output:07backward_simple_rnn_10/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp/^backward_simple_rnn_10/StatefulPartitionedCall.^forward_simple_rnn_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2`
.backward_simple_rnn_10/StatefulPartitionedCall.backward_simple_rnn_10/StatefulPartitionedCall2^
-forward_simple_rnn_10/StatefulPartitionedCall-forward_simple_rnn_10/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
ü
J__inference_sequential_21_layer_call_and_return_conditional_losses_5008846

inputs*
bidirectional_21_5008815:4@&
bidirectional_21_5008817:@*
bidirectional_21_5008819:@@*
bidirectional_21_5008821:4@&
bidirectional_21_5008823:@*
bidirectional_21_5008825:@@#
dense_21_5008840:	
dense_21_5008842:
identity¢(bidirectional_21/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall
(bidirectional_21/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_21_5008815bidirectional_21_5008817bidirectional_21_5008819bidirectional_21_5008821bidirectional_21_5008823bidirectional_21_5008825*
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5008814¡
 dense_21/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_21/StatefulPartitionedCall:output:0dense_21_5008840dense_21_5008842*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_5008839x
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_21/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_21/StatefulPartitionedCall(bidirectional_21/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ó-
Ò
while_body_5011324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_32_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_32/MatMul/ReadVariableOp¢0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_32/BiasAddBiasAdd)while/simple_rnn_cell_32/MatMul:product:07while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_32/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_32/addAddV2)while/simple_rnn_cell_32/BiasAdd:output:0+while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_32/TanhTanh while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_32/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_32/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_32/MatMul/ReadVariableOp1^while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_32_biasadd_readvariableop_resource:while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_32_matmul_1_readvariableop_resource;while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_32_matmul_readvariableop_resource9while_simple_rnn_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_32/MatMul/ReadVariableOp.while/simple_rnn_cell_32/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp0while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5011851

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
ÏC

)backward_simple_rnn_10_while_body_5008745J
Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counterP
Lbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations,
(backward_simple_rnn_10_while_placeholder.
*backward_simple_rnn_10_while_placeholder_1.
*backward_simple_rnn_10_while_placeholder_2I
Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_10_while_identity+
'backward_simple_rnn_10_while_identity_1+
'backward_simple_rnn_10_while_identity_2+
'backward_simple_rnn_10_while_identity_3+
'backward_simple_rnn_10_while_identity_4G
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp¢Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
@backward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_10_while_placeholderWbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ö
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_10/while/simple_rnn_cell_32/MatMulMatMulGbackward_simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAddBiasAdd@backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul:product:0Nbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1MatMul*backward_simple_rnn_10_while_placeholder_2Obackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_10/while/simple_rnn_cell_32/addAddV2@backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd:output:0Bbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_10/while/simple_rnn_cell_32/TanhTanh7backward_simple_rnn_10/while/simple_rnn_cell_32/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_10_while_placeholder_1Pbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_10/while/addAddV2(backward_simple_rnn_10_while_placeholder+backward_simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_10/while/add_1AddV2Fbackward_simple_rnn_10_while_backward_simple_rnn_10_while_loop_counter-backward_simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_10/while/IdentityIdentity&backward_simple_rnn_10/while/add_1:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_10/while/Identity_1IdentityLbackward_simple_rnn_10_while_backward_simple_rnn_10_while_maximum_iterations"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_10/while/Identity_2Identity$backward_simple_rnn_10/while/add:z:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_10/while/Identity_3IdentityQbackward_simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_10/while/Identity_4Identity8backward_simple_rnn_10/while/simple_rnn_cell_32/Tanh:y:0"^backward_simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_10/while/NoOpNoOpG^backward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpF^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpH^backward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1Ebackward_simple_rnn_10_while_backward_simple_rnn_10_strided_slice_1_0"W
%backward_simple_rnn_10_while_identity.backward_simple_rnn_10/while/Identity:output:0"[
'backward_simple_rnn_10_while_identity_10backward_simple_rnn_10/while/Identity_1:output:0"[
'backward_simple_rnn_10_while_identity_20backward_simple_rnn_10/while/Identity_2:output:0"[
'backward_simple_rnn_10_while_identity_30backward_simple_rnn_10/while/Identity_3:output:0"[
'backward_simple_rnn_10_while_identity_40backward_simple_rnn_10/while/Identity_4:output:0"¤
Obackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resourceQbackward_simple_rnn_10_while_simple_rnn_cell_32_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resourceRbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resourcePbackward_simple_rnn_10_while_simple_rnn_cell_32_matmul_readvariableop_resource_0"
backward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOpFbackward_simple_rnn_10/while/simple_rnn_cell_32/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOpEbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul/ReadVariableOp2
Gbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOpGbackward_simple_rnn_10/while/simple_rnn_cell_32/MatMul_1/ReadVariableOp: 
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

î
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5008270

inputs/
forward_simple_rnn_10_5008141:4@+
forward_simple_rnn_10_5008143:@/
forward_simple_rnn_10_5008145:@@0
backward_simple_rnn_10_5008260:4@,
backward_simple_rnn_10_5008262:@0
backward_simple_rnn_10_5008264:@@
identity¢.backward_simple_rnn_10/StatefulPartitionedCall¢-forward_simple_rnn_10/StatefulPartitionedCallË
-forward_simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_10_5008141forward_simple_rnn_10_5008143forward_simple_rnn_10_5008145*
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5008140Ð
.backward_simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_10_5008260backward_simple_rnn_10_5008262backward_simple_rnn_10_5008264*
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
GPU2*0J 8 *\
fWRU
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5008259M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Õ
concatConcatV26forward_simple_rnn_10/StatefulPartitionedCall:output:07backward_simple_rnn_10/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp/^backward_simple_rnn_10/StatefulPartitionedCall.^forward_simple_rnn_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2`
.backward_simple_rnn_10/StatefulPartitionedCall.backward_simple_rnn_10/StatefulPartitionedCall2^
-forward_simple_rnn_10/StatefulPartitionedCall-forward_simple_rnn_10/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø	
É
%__inference_signature_wrapper_5009287
bidirectional_21_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallbidirectional_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
"__inference__wrapped_model_5007428o
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
_user_specified_namebidirectional_21_input
ß
¯
while_cond_5010837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5010837___redundant_placeholder05
1while_while_cond_5010837___redundant_placeholder15
1while_while_cond_5010837___redundant_placeholder25
1while_while_cond_5010837___redundant_placeholder3
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
ä

J__inference_sequential_21_layer_call_and_return_conditional_losses_5009258
bidirectional_21_input*
bidirectional_21_5009239:4@&
bidirectional_21_5009241:@*
bidirectional_21_5009243:@@*
bidirectional_21_5009245:4@&
bidirectional_21_5009247:@*
bidirectional_21_5009249:@@#
dense_21_5009252:	
dense_21_5009254:
identity¢(bidirectional_21/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall
(bidirectional_21/StatefulPartitionedCallStatefulPartitionedCallbidirectional_21_inputbidirectional_21_5009239bidirectional_21_5009241bidirectional_21_5009243bidirectional_21_5009245bidirectional_21_5009247bidirectional_21_5009249*
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5009114¡
 dense_21/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_21/StatefulPartitionedCall:output:0dense_21_5009252dense_21_5009254*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_5008839x
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_21/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_21/StatefulPartitionedCall(bidirectional_21/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_21_input
¡
¯	
:bidirectional_21_backward_simple_rnn_10_while_cond_5009479l
hbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_loop_counterr
nbidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_maximum_iterations=
9bidirectional_21_backward_simple_rnn_10_while_placeholder?
;bidirectional_21_backward_simple_rnn_10_while_placeholder_1?
;bidirectional_21_backward_simple_rnn_10_while_placeholder_2n
jbidirectional_21_backward_simple_rnn_10_while_less_bidirectional_21_backward_simple_rnn_10_strided_slice_1
bidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_cond_5009479___redundant_placeholder0
bidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_cond_5009479___redundant_placeholder1
bidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_cond_5009479___redundant_placeholder2
bidirectional_21_backward_simple_rnn_10_while_bidirectional_21_backward_simple_rnn_10_while_cond_5009479___redundant_placeholder3:
6bidirectional_21_backward_simple_rnn_10_while_identity

2bidirectional_21/backward_simple_rnn_10/while/LessLess9bidirectional_21_backward_simple_rnn_10_while_placeholderjbidirectional_21_backward_simple_rnn_10_while_less_bidirectional_21_backward_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: 
6bidirectional_21/backward_simple_rnn_10/while/IdentityIdentity6bidirectional_21/backward_simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "y
6bidirectional_21_backward_simple_rnn_10_while_identity?bidirectional_21/backward_simple_rnn_10/while/Identity:output:0*(
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
:"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Í
serving_default¹
]
bidirectional_21_inputC
(serving_default_bidirectional_21_input:0ÿÿÿÿÿÿÿÿÿ4<
dense_210
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Õ¤
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
/__inference_sequential_21_layer_call_fn_5008865
/__inference_sequential_21_layer_call_fn_5009308
/__inference_sequential_21_layer_call_fn_5009329
/__inference_sequential_21_layer_call_fn_5009214¿
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009556
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009783
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009236
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009258¿
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
"__inference__wrapped_model_5007428bidirectional_21_input"
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
2__inference_bidirectional_21_layer_call_fn_5009800
2__inference_bidirectional_21_layer_call_fn_5009817
2__inference_bidirectional_21_layer_call_fn_5009834
2__inference_bidirectional_21_layer_call_fn_5009851å
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010071
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010291
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010511
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010731å
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
*__inference_dense_21_layer_call_fn_5010740¢
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
E__inference_dense_21_layer_call_and_return_conditional_losses_5010751¢
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
": 	2dense_21/kernel
:2dense_21/bias
R:P4@2@bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel
\:Z@@2Jbidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel
L:J@2>bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias
S:Q4@2Abidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel
]:[@@2Kbidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel
M:K@2?bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias
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
/__inference_sequential_21_layer_call_fn_5008865bidirectional_21_input"¿
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
/__inference_sequential_21_layer_call_fn_5009308inputs"¿
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
/__inference_sequential_21_layer_call_fn_5009329inputs"¿
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
/__inference_sequential_21_layer_call_fn_5009214bidirectional_21_input"¿
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009556inputs"¿
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009783inputs"¿
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009236bidirectional_21_input"¿
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009258bidirectional_21_input"¿
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
%__inference_signature_wrapper_5009287bidirectional_21_input"
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
2__inference_bidirectional_21_layer_call_fn_5009800inputs/0"å
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
2__inference_bidirectional_21_layer_call_fn_5009817inputs/0"å
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
2__inference_bidirectional_21_layer_call_fn_5009834inputs"å
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
2__inference_bidirectional_21_layer_call_fn_5009851inputs"å
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010071inputs/0"å
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010291inputs/0"å
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010511inputs"å
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010731inputs"å
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
¦
atrace_0
btrace_1
ctrace_2
dtrace_32»
7__inference_forward_simple_rnn_10_layer_call_fn_5010762
7__inference_forward_simple_rnn_10_layer_call_fn_5010773
7__inference_forward_simple_rnn_10_layer_call_fn_5010784
7__inference_forward_simple_rnn_10_layer_call_fn_5010795Ô
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

etrace_0
ftrace_1
gtrace_2
htrace_32§
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5010905
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011015
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011125
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011235Ô
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
ª
vtrace_0
wtrace_1
xtrace_2
ytrace_32¿
8__inference_backward_simple_rnn_10_layer_call_fn_5011246
8__inference_backward_simple_rnn_10_layer_call_fn_5011257
8__inference_backward_simple_rnn_10_layer_call_fn_5011268
8__inference_backward_simple_rnn_10_layer_call_fn_5011279Ô
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

ztrace_0
{trace_1
|trace_2
}trace_32«
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011391
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011503
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011615
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011727Ô
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
*__inference_dense_21_layer_call_fn_5010740inputs"¢
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
E__inference_dense_21_layer_call_and_return_conditional_losses_5010751inputs"¢
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
B
7__inference_forward_simple_rnn_10_layer_call_fn_5010762inputs/0"Ô
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
7__inference_forward_simple_rnn_10_layer_call_fn_5010773inputs/0"Ô
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
7__inference_forward_simple_rnn_10_layer_call_fn_5010784inputs"Ô
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
7__inference_forward_simple_rnn_10_layer_call_fn_5010795inputs"Ô
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5010905inputs/0"Ô
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011015inputs/0"Ô
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011125inputs"Ô
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011235inputs"Ô
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
4__inference_simple_rnn_cell_31_layer_call_fn_5011741
4__inference_simple_rnn_cell_31_layer_call_fn_5011755½
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5011772
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5011789½
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
 B
8__inference_backward_simple_rnn_10_layer_call_fn_5011246inputs/0"Ô
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
 B
8__inference_backward_simple_rnn_10_layer_call_fn_5011257inputs/0"Ô
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
8__inference_backward_simple_rnn_10_layer_call_fn_5011268inputs"Ô
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
8__inference_backward_simple_rnn_10_layer_call_fn_5011279inputs"Ô
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
»B¸
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011391inputs/0"Ô
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
»B¸
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011503inputs/0"Ô
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
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011615inputs"Ô
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
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011727inputs"Ô
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
4__inference_simple_rnn_cell_32_layer_call_fn_5011803
4__inference_simple_rnn_cell_32_layer_call_fn_5011817½
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5011834
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5011851½
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
4__inference_simple_rnn_cell_31_layer_call_fn_5011741inputsstates/0"½
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
4__inference_simple_rnn_cell_31_layer_call_fn_5011755inputsstates/0"½
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5011772inputsstates/0"½
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5011789inputsstates/0"½
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
4__inference_simple_rnn_cell_32_layer_call_fn_5011803inputsstates/0"½
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
4__inference_simple_rnn_cell_32_layer_call_fn_5011817inputsstates/0"½
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5011834inputsstates/0"½
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5011851inputsstates/0"½
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
':%	2Adam/dense_21/kernel/m
 :2Adam/dense_21/bias/m
W:U4@2GAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/m
a:_@@2QAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/m
Q:O@2EAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/m
X:V4@2HAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/m
b:`@@2RAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/m
R:P@2FAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/m
':%	2Adam/dense_21/kernel/v
 :2Adam/dense_21/bias/v
W:U4@2GAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/kernel/v
a:_@@2QAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/recurrent_kernel/v
Q:O@2EAdam/bidirectional_21/forward_simple_rnn_10/simple_rnn_cell_31/bias/v
X:V4@2HAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/kernel/v
b:`@@2RAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/recurrent_kernel/v
R:P@2FAdam/bidirectional_21/backward_simple_rnn_10/simple_rnn_cell_32/bias/v«
"__inference__wrapped_model_5007428! C¢@
9¢6
41
bidirectional_21_inputÿÿÿÿÿÿÿÿÿ4
ª "3ª0
.
dense_21"
dense_21ÿÿÿÿÿÿÿÿÿÔ
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011391}! O¢L
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
 Ô
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011503}! O¢L
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
 Ö
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011615! Q¢N
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
 Ö
S__inference_backward_simple_rnn_10_layer_call_and_return_conditional_losses_5011727! Q¢N
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
 ¬
8__inference_backward_simple_rnn_10_layer_call_fn_5011246p! O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@¬
8__inference_backward_simple_rnn_10_layer_call_fn_5011257p! O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@®
8__inference_backward_simple_rnn_10_layer_call_fn_5011268r! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@®
8__inference_backward_simple_rnn_10_layer_call_fn_5011279r! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@à
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010071! \¢Y
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010291! \¢Y
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010511u! C¢@
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
M__inference_bidirectional_21_layer_call_and_return_conditional_losses_5010731u! C¢@
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
2__inference_bidirectional_21_layer_call_fn_5009800! \¢Y
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
2__inference_bidirectional_21_layer_call_fn_5009817! \¢Y
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
2__inference_bidirectional_21_layer_call_fn_5009834h! C¢@
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
2__inference_bidirectional_21_layer_call_fn_5009851h! C¢@
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
E__inference_dense_21_layer_call_and_return_conditional_losses_5010751]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_dense_21_layer_call_fn_5010740P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÓ
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5010905}O¢L
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011015}O¢L
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011125Q¢N
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
R__inference_forward_simple_rnn_10_layer_call_and_return_conditional_losses_5011235Q¢N
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
7__inference_forward_simple_rnn_10_layer_call_fn_5010762pO¢L
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
7__inference_forward_simple_rnn_10_layer_call_fn_5010773pO¢L
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
7__inference_forward_simple_rnn_10_layer_call_fn_5010784rQ¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@­
7__inference_forward_simple_rnn_10_layer_call_fn_5010795rQ¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@Ì
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009236~! K¢H
A¢>
41
bidirectional_21_inputÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009258~! K¢H
A¢>
41
bidirectional_21_inputÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009556n! ;¢8
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_5009783n! ;¢8
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
/__inference_sequential_21_layer_call_fn_5008865q! K¢H
A¢>
41
bidirectional_21_inputÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
/__inference_sequential_21_layer_call_fn_5009214q! K¢H
A¢>
41
bidirectional_21_inputÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_21_layer_call_fn_5009308a! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_21_layer_call_fn_5009329a! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ
%__inference_signature_wrapper_5009287! ]¢Z
¢ 
SªP
N
bidirectional_21_input41
bidirectional_21_inputÿÿÿÿÿÿÿÿÿ4"3ª0
.
dense_21"
dense_21ÿÿÿÿÿÿÿÿÿ
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5011772·\¢Y
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
O__inference_simple_rnn_cell_31_layer_call_and_return_conditional_losses_5011789·\¢Y
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
4__inference_simple_rnn_cell_31_layer_call_fn_5011741©\¢Y
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
4__inference_simple_rnn_cell_31_layer_call_fn_5011755©\¢Y
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5011834·! \¢Y
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
O__inference_simple_rnn_cell_32_layer_call_and_return_conditional_losses_5011851·! \¢Y
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
4__inference_simple_rnn_cell_32_layer_call_fn_5011803©! \¢Y
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
4__inference_simple_rnn_cell_32_layer_call_fn_5011817©! \¢Y
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