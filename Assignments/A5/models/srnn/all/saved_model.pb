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
FAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*W
shared_nameHFAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/v
Ý
ZAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/v/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/v*
_output_shapes
:@*
dtype0

RAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*c
shared_nameTRAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/v
ù
fAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpRAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
ì
HAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*Y
shared_nameJHAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/v
å
\Adam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/v/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/v*
_output_shapes

:4@*
dtype0
â
EAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*V
shared_nameGEAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/v
Û
YAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/v/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/v*
_output_shapes
:@*
dtype0
þ
QAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*b
shared_nameSQAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/v
÷
eAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpQAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
ê
GAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*X
shared_nameIGAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/v
ã
[Adam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/v*
_output_shapes

:4@*
dtype0

Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:*
dtype0

Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_24/kernel/v

*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes
:	*
dtype0
ä
FAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*W
shared_nameHFAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/m
Ý
ZAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/m/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/m*
_output_shapes
:@*
dtype0

RAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*c
shared_nameTRAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/m
ù
fAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpRAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
ì
HAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*Y
shared_nameJHAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/m
å
\Adam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/m/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/m*
_output_shapes

:4@*
dtype0
â
EAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*V
shared_nameGEAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/m
Û
YAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/m/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/m*
_output_shapes
:@*
dtype0
þ
QAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*b
shared_nameSQAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/m
÷
eAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpQAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
ê
GAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*X
shared_nameIGAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/m
ã
[Adam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/m*
_output_shapes

:4@*
dtype0

Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:*
dtype0

Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_24/kernel/m

*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
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
?bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias
Ï
Sbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/Read/ReadVariableOpReadVariableOp?bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias*
_output_shapes
:@*
dtype0
ò
Kbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*\
shared_nameMKbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel
ë
_bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/Read/ReadVariableOpReadVariableOpKbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel*
_output_shapes

:@@*
dtype0
Þ
Abidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*R
shared_nameCAbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel
×
Ubidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/Read/ReadVariableOpReadVariableOpAbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel*
_output_shapes

:4@*
dtype0
Ô
>bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*O
shared_name@>bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias
Í
Rbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/Read/ReadVariableOpReadVariableOp>bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias*
_output_shapes
:@*
dtype0
ð
Jbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*[
shared_nameLJbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel
é
^bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/Read/ReadVariableOpReadVariableOpJbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel*
_output_shapes

:@@*
dtype0
Ü
@bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4@*Q
shared_nameB@bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel
Õ
Tbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/Read/ReadVariableOpReadVariableOp@bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel*
_output_shapes

:4@*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:*
dtype0
{
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_24/kernel
t
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes
:	*
dtype0

&serving_default_bidirectional_24_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ4

StatefulPartitionedCallStatefulPartitionedCall&serving_default_bidirectional_24_input@bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel>bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/biasJbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernelAbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel?bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/biasKbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kerneldense_24/kerneldense_24/bias*
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
%__inference_signature_wrapper_5422521

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
VARIABLE_VALUEdense_24/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_24/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE@bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEKbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¥
VARIABLE_VALUEHAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¯¨
VARIABLE_VALUERAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
£
VARIABLE_VALUEFAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¥
VARIABLE_VALUEHAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¯¨
VARIABLE_VALUERAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
£
VARIABLE_VALUEFAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
æ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOpTbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/Read/ReadVariableOp^bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/Read/ReadVariableOpRbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/Read/ReadVariableOpUbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/Read/ReadVariableOp_bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/Read/ReadVariableOpSbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp[Adam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/m/Read/ReadVariableOpeAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/m/Read/ReadVariableOpYAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/m/Read/ReadVariableOp\Adam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/m/Read/ReadVariableOpfAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/m/Read/ReadVariableOpZAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp[Adam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/v/Read/ReadVariableOpeAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/v/Read/ReadVariableOpYAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/v/Read/ReadVariableOp\Adam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/v/Read/ReadVariableOpfAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/v/Read/ReadVariableOpZAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/v/Read/ReadVariableOpConst*.
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
 __inference__traced_save_5425207
Í
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_24/kerneldense_24/bias@bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernelJbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel>bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/biasAbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernelKbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel?bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_24/kernel/mAdam/dense_24/bias/mGAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/mQAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/mEAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/mHAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/mRAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/mFAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/mAdam/dense_24/kernel/vAdam/dense_24/bias/vGAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/vQAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/vEAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/vHAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/vRAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/vFAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/v*-
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
#__inference__traced_restore_5425316·õ,
¢Ó
ý
J__inference_sequential_24_layer_call_and_return_conditional_losses_5423017

inputsj
Xbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource:4@g
Ybidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource:@l
Zbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@k
Ybidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource:4@h
Zbidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource:@m
[bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@:
'dense_24_matmul_readvariableop_resource:	6
(dense_24_biasadd_readvariableop_resource:
identity¢Qbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Pbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp¢Rbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp¢-bidirectional_24/backward_simple_rnn_11/while¢Pbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Obidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp¢Qbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp¢,bidirectional_24/forward_simple_rnn_11/while¢dense_24/BiasAdd/ReadVariableOp¢dense_24/MatMul/ReadVariableOpb
,bidirectional_24/forward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:
:bidirectional_24/forward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<bidirectional_24/forward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<bidirectional_24/forward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4bidirectional_24/forward_simple_rnn_11/strided_sliceStridedSlice5bidirectional_24/forward_simple_rnn_11/Shape:output:0Cbidirectional_24/forward_simple_rnn_11/strided_slice/stack:output:0Ebidirectional_24/forward_simple_rnn_11/strided_slice/stack_1:output:0Ebidirectional_24/forward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5bidirectional_24/forward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@è
3bidirectional_24/forward_simple_rnn_11/zeros/packedPack=bidirectional_24/forward_simple_rnn_11/strided_slice:output:0>bidirectional_24/forward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:w
2bidirectional_24/forward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    á
,bidirectional_24/forward_simple_rnn_11/zerosFill<bidirectional_24/forward_simple_rnn_11/zeros/packed:output:0;bidirectional_24/forward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5bidirectional_24/forward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
0bidirectional_24/forward_simple_rnn_11/transpose	Transposeinputs>bidirectional_24/forward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
.bidirectional_24/forward_simple_rnn_11/Shape_1Shape4bidirectional_24/forward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:
<bidirectional_24/forward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_24/forward_simple_rnn_11/strided_slice_1StridedSlice7bidirectional_24/forward_simple_rnn_11/Shape_1:output:0Ebidirectional_24/forward_simple_rnn_11/strided_slice_1/stack:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_1:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bbidirectional_24/forward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
4bidirectional_24/forward_simple_rnn_11/TensorArrayV2TensorListReserveKbidirectional_24/forward_simple_rnn_11/TensorArrayV2/element_shape:output:0?bidirectional_24/forward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ­
\bidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Õ
Nbidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor4bidirectional_24/forward_simple_rnn_11/transpose:y:0ebidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<bidirectional_24/forward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
6bidirectional_24/forward_simple_rnn_11/strided_slice_2StridedSlice4bidirectional_24/forward_simple_rnn_11/transpose:y:0Ebidirectional_24/forward_simple_rnn_11/strided_slice_2/stack:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_1:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskè
Obidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpXbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
@bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMulMatMul?bidirectional_24/forward_simple_rnn_11/strided_slice_2:output:0Wbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Pbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpYbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Abidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAddBiasAddJbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul:product:0Xbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ì
Qbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpZbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Bbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1MatMul5bidirectional_24/forward_simple_rnn_11/zeros:output:0Ybidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
=bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/addAddV2Jbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd:output:0Lbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
>bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/TanhTanhAbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dbidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Cbidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :º
6bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1TensorListReserveMbidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0Lbidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
+bidirectional_24/forward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 
?bidirectional_24/forward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ{
9bidirectional_24/forward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø	
,bidirectional_24/forward_simple_rnn_11/whileWhileBbidirectional_24/forward_simple_rnn_11/while/loop_counter:output:0Hbidirectional_24/forward_simple_rnn_11/while/maximum_iterations:output:04bidirectional_24/forward_simple_rnn_11/time:output:0?bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1:handle:05bidirectional_24/forward_simple_rnn_11/zeros:output:0?bidirectional_24/forward_simple_rnn_11/strided_slice_1:output:0^bidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Xbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resourceYbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resourceZbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
9bidirectional_24_forward_simple_rnn_11_while_body_5422833*E
cond=R;
9bidirectional_24_forward_simple_rnn_11_while_cond_5422832*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¨
Wbidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
Ibidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack5bidirectional_24/forward_simple_rnn_11/while:output:3`bidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
<bidirectional_24/forward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
>bidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ê
6bidirectional_24/forward_simple_rnn_11/strided_slice_3StridedSliceRbidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Ebidirectional_24/forward_simple_rnn_11/strided_slice_3/stack:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_1:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
7bidirectional_24/forward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
2bidirectional_24/forward_simple_rnn_11/transpose_1	TransposeRbidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional_24/forward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
-bidirectional_24/backward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:
;bidirectional_24/backward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_24/backward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_24/backward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_24/backward_simple_rnn_11/strided_sliceStridedSlice6bidirectional_24/backward_simple_rnn_11/Shape:output:0Dbidirectional_24/backward_simple_rnn_11/strided_slice/stack:output:0Fbidirectional_24/backward_simple_rnn_11/strided_slice/stack_1:output:0Fbidirectional_24/backward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6bidirectional_24/backward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@ë
4bidirectional_24/backward_simple_rnn_11/zeros/packedPack>bidirectional_24/backward_simple_rnn_11/strided_slice:output:0?bidirectional_24/backward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:x
3bidirectional_24/backward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ä
-bidirectional_24/backward_simple_rnn_11/zerosFill=bidirectional_24/backward_simple_rnn_11/zeros/packed:output:0<bidirectional_24/backward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
6bidirectional_24/backward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
1bidirectional_24/backward_simple_rnn_11/transpose	Transposeinputs?bidirectional_24/backward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
/bidirectional_24/backward_simple_rnn_11/Shape_1Shape5bidirectional_24/backward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:
=bidirectional_24/backward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?bidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7bidirectional_24/backward_simple_rnn_11/strided_slice_1StridedSlice8bidirectional_24/backward_simple_rnn_11/Shape_1:output:0Fbidirectional_24/backward_simple_rnn_11/strided_slice_1/stack:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_1:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Cbidirectional_24/backward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¬
5bidirectional_24/backward_simple_rnn_11/TensorArrayV2TensorListReserveLbidirectional_24/backward_simple_rnn_11/TensorArrayV2/element_shape:output:0@bidirectional_24/backward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
6bidirectional_24/backward_simple_rnn_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ì
1bidirectional_24/backward_simple_rnn_11/ReverseV2	ReverseV25bidirectional_24/backward_simple_rnn_11/transpose:y:0?bidirectional_24/backward_simple_rnn_11/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4®
]bidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ý
Obidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor:bidirectional_24/backward_simple_rnn_11/ReverseV2:output:0fbidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=bidirectional_24/backward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?bidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:±
7bidirectional_24/backward_simple_rnn_11/strided_slice_2StridedSlice5bidirectional_24/backward_simple_rnn_11/transpose:y:0Fbidirectional_24/backward_simple_rnn_11/strided_slice_2/stack:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_1:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskê
Pbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpYbidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
Abidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMulMatMul@bidirectional_24/backward_simple_rnn_11/strided_slice_2:output:0Xbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@è
Qbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpZbidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
Bbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAddBiasAddKbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul:product:0Ybidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
Rbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp[bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Cbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1MatMul6bidirectional_24/backward_simple_rnn_11/zeros:output:0Zbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/addAddV2Kbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd:output:0Mbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@½
?bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/TanhTanhBbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Ebidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Dbidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :½
7bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1TensorListReserveNbidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0Mbidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
,bidirectional_24/backward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 
@bidirectional_24/backward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ|
:bidirectional_24/backward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : å	
-bidirectional_24/backward_simple_rnn_11/whileWhileCbidirectional_24/backward_simple_rnn_11/while/loop_counter:output:0Ibidirectional_24/backward_simple_rnn_11/while/maximum_iterations:output:05bidirectional_24/backward_simple_rnn_11/time:output:0@bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1:handle:06bidirectional_24/backward_simple_rnn_11/zeros:output:0@bidirectional_24/backward_simple_rnn_11/strided_slice_1:output:0_bidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ybidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resourceZbidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource[bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
:bidirectional_24_backward_simple_rnn_11_while_body_5422941*F
cond>R<
:bidirectional_24_backward_simple_rnn_11_while_cond_5422940*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ©
Xbidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Î
Jbidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack6bidirectional_24/backward_simple_rnn_11/while:output:3abidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
=bidirectional_24/backward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
?bidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
7bidirectional_24/backward_simple_rnn_11/strided_slice_3StridedSliceSbidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Fbidirectional_24/backward_simple_rnn_11/strided_slice_3/stack:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_1:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
8bidirectional_24/backward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
3bidirectional_24/backward_simple_rnn_11/transpose_1	TransposeSbidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Abidirectional_24/backward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
bidirectional_24/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
bidirectional_24/concatConcatV2?bidirectional_24/forward_simple_rnn_11/strided_slice_3:output:0@bidirectional_24/backward_simple_rnn_11/strided_slice_3:output:0%bidirectional_24/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_24/MatMulMatMul bidirectional_24/concat:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_24/SoftmaxSoftmaxdense_24/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_24/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOpR^bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpQ^bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpS^bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp.^bidirectional_24/backward_simple_rnn_11/whileQ^bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpP^bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpR^bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp-^bidirectional_24/forward_simple_rnn_11/while ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2¦
Qbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpQbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp2¤
Pbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpPbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp2¨
Rbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpRbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp2^
-bidirectional_24/backward_simple_rnn_11/while-bidirectional_24/backward_simple_rnn_11/while2¤
Pbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpPbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp2¢
Obidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpObidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp2¦
Qbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpQbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp2\
,bidirectional_24/forward_simple_rnn_11/while,bidirectional_24/forward_simple_rnn_11/while2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ß
¯
while_cond_5424669
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5424669___redundant_placeholder05
1while_while_cond_5424669___redundant_placeholder15
1while_while_cond_5424669___redundant_placeholder25
1while_while_cond_5424669___redundant_placeholder3
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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5420710

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
?
Ë
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5421374

inputsC
1simple_rnn_cell_34_matmul_readvariableop_resource:4@@
2simple_rnn_cell_34_biasadd_readvariableop_resource:@E
3simple_rnn_cell_34_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_34/BiasAdd/ReadVariableOp¢(simple_rnn_cell_34/MatMul/ReadVariableOp¢*simple_rnn_cell_34/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_34/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_34/BiasAddBiasAdd#simple_rnn_cell_34/MatMul:product:01simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_34/MatMul_1MatMulzeros:output:02simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_34/addAddV2#simple_rnn_cell_34/BiasAdd:output:0%simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_34/TanhTanhsimple_rnn_cell_34/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_34_matmul_readvariableop_resource2simple_rnn_cell_34_biasadd_readvariableop_resource3simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
while_body_5421307*
condR
while_cond_5421306*8
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
NoOpNoOp*^simple_rnn_cell_34/BiasAdd/ReadVariableOp)^simple_rnn_cell_34/MatMul/ReadVariableOp+^simple_rnn_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_34/BiasAdd/ReadVariableOp)simple_rnn_cell_34/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_34/MatMul/ReadVariableOp(simple_rnn_cell_34/MatMul/ReadVariableOp2X
*simple_rnn_cell_34/MatMul_1/ReadVariableOp*simple_rnn_cell_34/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
¯	
:bidirectional_24_backward_simple_rnn_11_while_cond_5422713l
hbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_loop_counterr
nbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_maximum_iterations=
9bidirectional_24_backward_simple_rnn_11_while_placeholder?
;bidirectional_24_backward_simple_rnn_11_while_placeholder_1?
;bidirectional_24_backward_simple_rnn_11_while_placeholder_2n
jbidirectional_24_backward_simple_rnn_11_while_less_bidirectional_24_backward_simple_rnn_11_strided_slice_1
bidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_cond_5422713___redundant_placeholder0
bidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_cond_5422713___redundant_placeholder1
bidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_cond_5422713___redundant_placeholder2
bidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_cond_5422713___redundant_placeholder3:
6bidirectional_24_backward_simple_rnn_11_while_identity

2bidirectional_24/backward_simple_rnn_11/while/LessLess9bidirectional_24_backward_simple_rnn_11_while_placeholderjbidirectional_24_backward_simple_rnn_11_while_less_bidirectional_24_backward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: 
6bidirectional_24/backward_simple_rnn_11/while/IdentityIdentity6bidirectional_24/backward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "y
6bidirectional_24_backward_simple_rnn_11_while_identity?bidirectional_24/backward_simple_rnn_11/while/Identity:output:0*(
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
þ¨
Û
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423965

inputsY
Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp¢Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp¢backward_simple_rnn_11/while¢?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp¢@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp¢forward_simple_rnn_11/whileQ
forward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:s
)forward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_11/strided_sliceStridedSlice$forward_simple_rnn_11/Shape:output:02forward_simple_rnn_11/strided_slice/stack:output:04forward_simple_rnn_11/strided_slice/stack_1:output:04forward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_11/zeros/packedPack,forward_simple_rnn_11/strided_slice:output:0-forward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_11/zerosFill+forward_simple_rnn_11/zeros/packed:output:0*forward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_11/transpose	Transposeinputs-forward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
forward_simple_rnn_11/Shape_1Shape#forward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_11/strided_slice_1StridedSlice&forward_simple_rnn_11/Shape_1:output:04forward_simple_rnn_11/strided_slice_1/stack:output:06forward_simple_rnn_11/strided_slice_1/stack_1:output:06forward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_11/TensorArrayV2TensorListReserve:forward_simple_rnn_11/TensorArrayV2/element_shape:output:0.forward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¢
=forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_11/transpose:y:0Tforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%forward_simple_rnn_11/strided_slice_2StridedSlice#forward_simple_rnn_11/transpose:y:04forward_simple_rnn_11/strided_slice_2/stack:output:06forward_simple_rnn_11/strided_slice_2/stack_1:output:06forward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_11/simple_rnn_cell_34/MatMulMatMul.forward_simple_rnn_11/strided_slice_2:output:0Fforward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_11/simple_rnn_cell_34/BiasAddBiasAdd9forward_simple_rnn_11/simple_rnn_cell_34/MatMul:product:0Gforward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1MatMul$forward_simple_rnn_11/zeros:output:0Hforward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_11/simple_rnn_cell_34/addAddV29forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd:output:0;forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_11/simple_rnn_cell_34/TanhTanh0forward_simple_rnn_11/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_11/TensorArrayV2_1TensorListReserve<forward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_11/whileWhile1forward_simple_rnn_11/while/loop_counter:output:07forward_simple_rnn_11/while/maximum_iterations:output:0#forward_simple_rnn_11/time:output:0.forward_simple_rnn_11/TensorArrayV2_1:handle:0$forward_simple_rnn_11/zeros:output:0.forward_simple_rnn_11/strided_slice_1:output:0Mforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resourceHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resourceIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_11_while_body_5423788*4
cond,R*
(forward_simple_rnn_11_while_cond_5423787*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_11/while:output:3Oforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_11/strided_slice_3StridedSliceAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_11/strided_slice_3/stack:output:06forward_simple_rnn_11/strided_slice_3/stack_1:output:06forward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_11/transpose_1	TransposeAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
backward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:t
*backward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_11/strided_sliceStridedSlice%backward_simple_rnn_11/Shape:output:03backward_simple_rnn_11/strided_slice/stack:output:05backward_simple_rnn_11/strided_slice/stack_1:output:05backward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_11/zeros/packedPack-backward_simple_rnn_11/strided_slice:output:0.backward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_11/zerosFill,backward_simple_rnn_11/zeros/packed:output:0+backward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
 backward_simple_rnn_11/transpose	Transposeinputs.backward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4r
backward_simple_rnn_11/Shape_1Shape$backward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_11/strided_slice_1StridedSlice'backward_simple_rnn_11/Shape_1:output:05backward_simple_rnn_11/strided_slice_1/stack:output:07backward_simple_rnn_11/strided_slice_1/stack_1:output:07backward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_11/TensorArrayV2TensorListReserve;backward_simple_rnn_11/TensorArrayV2/element_shape:output:0/backward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¹
 backward_simple_rnn_11/ReverseV2	ReverseV2$backward_simple_rnn_11/transpose:y:0.backward_simple_rnn_11/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Lbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ª
>backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_11/ReverseV2:output:0Ubackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
&backward_simple_rnn_11/strided_slice_2StridedSlice$backward_simple_rnn_11/transpose:y:05backward_simple_rnn_11/strided_slice_2/stack:output:07backward_simple_rnn_11/strided_slice_2/stack_1:output:07backward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÈ
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_11/simple_rnn_cell_35/MatMulMatMul/backward_simple_rnn_11/strided_slice_2:output:0Gbackward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_11/simple_rnn_cell_35/BiasAddBiasAdd:backward_simple_rnn_11/simple_rnn_cell_35/MatMul:product:0Hbackward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1MatMul%backward_simple_rnn_11/zeros:output:0Ibackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_11/simple_rnn_cell_35/addAddV2:backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd:output:0<backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_11/simple_rnn_cell_35/TanhTanh1backward_simple_rnn_11/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_11/TensorArrayV2_1TensorListReserve=backward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_11/whileWhile2backward_simple_rnn_11/while/loop_counter:output:08backward_simple_rnn_11/while/maximum_iterations:output:0$backward_simple_rnn_11/time:output:0/backward_simple_rnn_11/TensorArrayV2_1:handle:0%backward_simple_rnn_11/zeros:output:0/backward_simple_rnn_11/strided_slice_1:output:0Nbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resourceIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resourceJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
)backward_simple_rnn_11_while_body_5423896*5
cond-R+
)backward_simple_rnn_11_while_cond_5423895*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_11/while:output:3Pbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_11/strided_slice_3StridedSliceBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_11/strided_slice_3/stack:output:07backward_simple_rnn_11/strided_slice_3/stack_1:output:07backward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_11/transpose_1	TransposeBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_11/strided_slice_3:output:0/backward_simple_rnn_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@^backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpB^backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp^backward_simple_rnn_11/while@^forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?^forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpA^forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp^forward_simple_rnn_11/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp2
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpAbackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp2<
backward_simple_rnn_11/whilebackward_simple_rnn_11/while2
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp2
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp2:
forward_simple_rnn_11/whileforward_simple_rnn_11/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
Í
ä
)backward_simple_rnn_11_while_cond_5423675J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2L
Hbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423675___redundant_placeholder0c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423675___redundant_placeholder1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423675___redundant_placeholder2c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423675___redundant_placeholder3)
%backward_simple_rnn_11_while_identity
¾
!backward_simple_rnn_11/while/LessLess(backward_simple_rnn_11_while_placeholderHbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_11/while/IdentityIdentity%backward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0*(
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
4__inference_simple_rnn_cell_34_layer_call_fn_5424989

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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5420832o
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
­
Ã
7__inference_forward_simple_rnn_11_layer_call_fn_5424007
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5420949o
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
þ¨
Û
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423745

inputsY
Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp¢Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp¢backward_simple_rnn_11/while¢?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp¢@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp¢forward_simple_rnn_11/whileQ
forward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:s
)forward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_11/strided_sliceStridedSlice$forward_simple_rnn_11/Shape:output:02forward_simple_rnn_11/strided_slice/stack:output:04forward_simple_rnn_11/strided_slice/stack_1:output:04forward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_11/zeros/packedPack,forward_simple_rnn_11/strided_slice:output:0-forward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_11/zerosFill+forward_simple_rnn_11/zeros/packed:output:0*forward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_11/transpose	Transposeinputs-forward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
forward_simple_rnn_11/Shape_1Shape#forward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_11/strided_slice_1StridedSlice&forward_simple_rnn_11/Shape_1:output:04forward_simple_rnn_11/strided_slice_1/stack:output:06forward_simple_rnn_11/strided_slice_1/stack_1:output:06forward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_11/TensorArrayV2TensorListReserve:forward_simple_rnn_11/TensorArrayV2/element_shape:output:0.forward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¢
=forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_11/transpose:y:0Tforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%forward_simple_rnn_11/strided_slice_2StridedSlice#forward_simple_rnn_11/transpose:y:04forward_simple_rnn_11/strided_slice_2/stack:output:06forward_simple_rnn_11/strided_slice_2/stack_1:output:06forward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_11/simple_rnn_cell_34/MatMulMatMul.forward_simple_rnn_11/strided_slice_2:output:0Fforward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_11/simple_rnn_cell_34/BiasAddBiasAdd9forward_simple_rnn_11/simple_rnn_cell_34/MatMul:product:0Gforward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1MatMul$forward_simple_rnn_11/zeros:output:0Hforward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_11/simple_rnn_cell_34/addAddV29forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd:output:0;forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_11/simple_rnn_cell_34/TanhTanh0forward_simple_rnn_11/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_11/TensorArrayV2_1TensorListReserve<forward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_11/whileWhile1forward_simple_rnn_11/while/loop_counter:output:07forward_simple_rnn_11/while/maximum_iterations:output:0#forward_simple_rnn_11/time:output:0.forward_simple_rnn_11/TensorArrayV2_1:handle:0$forward_simple_rnn_11/zeros:output:0.forward_simple_rnn_11/strided_slice_1:output:0Mforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resourceHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resourceIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_11_while_body_5423568*4
cond,R*
(forward_simple_rnn_11_while_cond_5423567*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_11/while:output:3Oforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_11/strided_slice_3StridedSliceAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_11/strided_slice_3/stack:output:06forward_simple_rnn_11/strided_slice_3/stack_1:output:06forward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_11/transpose_1	TransposeAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
backward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:t
*backward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_11/strided_sliceStridedSlice%backward_simple_rnn_11/Shape:output:03backward_simple_rnn_11/strided_slice/stack:output:05backward_simple_rnn_11/strided_slice/stack_1:output:05backward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_11/zeros/packedPack-backward_simple_rnn_11/strided_slice:output:0.backward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_11/zerosFill,backward_simple_rnn_11/zeros/packed:output:0+backward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
 backward_simple_rnn_11/transpose	Transposeinputs.backward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4r
backward_simple_rnn_11/Shape_1Shape$backward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_11/strided_slice_1StridedSlice'backward_simple_rnn_11/Shape_1:output:05backward_simple_rnn_11/strided_slice_1/stack:output:07backward_simple_rnn_11/strided_slice_1/stack_1:output:07backward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_11/TensorArrayV2TensorListReserve;backward_simple_rnn_11/TensorArrayV2/element_shape:output:0/backward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¹
 backward_simple_rnn_11/ReverseV2	ReverseV2$backward_simple_rnn_11/transpose:y:0.backward_simple_rnn_11/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Lbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ª
>backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_11/ReverseV2:output:0Ubackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
&backward_simple_rnn_11/strided_slice_2StridedSlice$backward_simple_rnn_11/transpose:y:05backward_simple_rnn_11/strided_slice_2/stack:output:07backward_simple_rnn_11/strided_slice_2/stack_1:output:07backward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÈ
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_11/simple_rnn_cell_35/MatMulMatMul/backward_simple_rnn_11/strided_slice_2:output:0Gbackward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_11/simple_rnn_cell_35/BiasAddBiasAdd:backward_simple_rnn_11/simple_rnn_cell_35/MatMul:product:0Hbackward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1MatMul%backward_simple_rnn_11/zeros:output:0Ibackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_11/simple_rnn_cell_35/addAddV2:backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd:output:0<backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_11/simple_rnn_cell_35/TanhTanh1backward_simple_rnn_11/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_11/TensorArrayV2_1TensorListReserve=backward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_11/whileWhile2backward_simple_rnn_11/while/loop_counter:output:08backward_simple_rnn_11/while/maximum_iterations:output:0$backward_simple_rnn_11/time:output:0/backward_simple_rnn_11/TensorArrayV2_1:handle:0%backward_simple_rnn_11/zeros:output:0/backward_simple_rnn_11/strided_slice_1:output:0Nbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resourceIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resourceJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
)backward_simple_rnn_11_while_body_5423676*5
cond-R+
)backward_simple_rnn_11_while_cond_5423675*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_11/while:output:3Pbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_11/strided_slice_3StridedSliceBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_11/strided_slice_3/stack:output:07backward_simple_rnn_11/strided_slice_3/stack_1:output:07backward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_11/transpose_1	TransposeBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_11/strided_slice_3:output:0/backward_simple_rnn_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@^backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpB^backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp^backward_simple_rnn_11/while@^forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?^forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpA^forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp^forward_simple_rnn_11/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp2
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpAbackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp2<
backward_simple_rnn_11/whilebackward_simple_rnn_11/while2
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp2
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp2:
forward_simple_rnn_11/whileforward_simple_rnn_11/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
²
Ñ
(forward_simple_rnn_11_while_cond_5423127H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2J
Fforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423127___redundant_placeholder0a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423127___redundant_placeholder1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423127___redundant_placeholder2a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423127___redundant_placeholder3(
$forward_simple_rnn_11_while_identity
º
 forward_simple_rnn_11/while/LessLess'forward_simple_rnn_11_while_placeholderFforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_11/while/IdentityIdentity$forward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0*(
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
(forward_simple_rnn_11_while_body_5423348H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2G
Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0
forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_11_while_identity*
&forward_simple_rnn_11_while_identity_1*
&forward_simple_rnn_11_while_identity_2*
&forward_simple_rnn_11_while_identity_3*
&forward_simple_rnn_11_while_identity_4E
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp¢Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
Mforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
?forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_11_while_placeholderVforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ô
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_11/while/simple_rnn_cell_34/MatMulMatMulFforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAddBiasAdd?forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul:product:0Mforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1MatMul)forward_simple_rnn_11_while_placeholder_2Nforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_11/while/simple_rnn_cell_34/addAddV2?forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd:output:0Aforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_11/while/simple_rnn_cell_34/TanhTanh6forward_simple_rnn_11/while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_11_while_placeholder_1Oforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_11/while/addAddV2'forward_simple_rnn_11_while_placeholder*forward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_11/while/add_1AddV2Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counter,forward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_11/while/IdentityIdentity%forward_simple_rnn_11/while/add_1:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_11/while/Identity_1IdentityJforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_11/while/Identity_2Identity#forward_simple_rnn_11/while/add:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_11/while/Identity_3IdentityPforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_11/while/Identity_4Identity7forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0!^forward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_11/while/NoOpNoOpF^forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpE^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpG^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0"U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0"Y
&forward_simple_rnn_11_while_identity_1/forward_simple_rnn_11/while/Identity_1:output:0"Y
&forward_simple_rnn_11_while_identity_2/forward_simple_rnn_11/while/Identity_2:output:0"Y
&forward_simple_rnn_11_while_identity_3/forward_simple_rnn_11/while/Identity_3:output:0"Y
&forward_simple_rnn_11_while_identity_4/forward_simple_rnn_11/while/Identity_4:output:0"¢
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resourcePforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resourceQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resourceOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0"
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpEforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpDforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp2
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpFforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
while_body_5421307
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_34_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_34/MatMul/ReadVariableOp¢0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_34/BiasAddBiasAdd)while/simple_rnn_cell_34/MatMul:product:07while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_34/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_34/addAddV2)while/simple_rnn_cell_34/BiasAdd:output:0+while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_34/TanhTanh while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_34/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_34/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_34/MatMul/ReadVariableOp1^while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_34_matmul_readvariableop_resource9while_simple_rnn_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_34/MatMul/ReadVariableOp.while/simple_rnn_cell_34/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
2__inference_bidirectional_24_layer_call_fn_5423068

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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5422048p
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
ÏC

)backward_simple_rnn_11_while_body_5423676J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2I
Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_11_while_identity+
'backward_simple_rnn_11_while_identity_1+
'backward_simple_rnn_11_while_identity_2+
'backward_simple_rnn_11_while_identity_3+
'backward_simple_rnn_11_while_identity_4G
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp¢Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
@backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_11_while_placeholderWbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ö
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_11/while/simple_rnn_cell_35/MatMulMatMulGbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAddBiasAdd@backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul:product:0Nbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1MatMul*backward_simple_rnn_11_while_placeholder_2Obackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_11/while/simple_rnn_cell_35/addAddV2@backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd:output:0Bbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_11/while/simple_rnn_cell_35/TanhTanh7backward_simple_rnn_11/while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_11_while_placeholder_1Pbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_11/while/addAddV2(backward_simple_rnn_11_while_placeholder+backward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_11/while/add_1AddV2Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counter-backward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_11/while/IdentityIdentity&backward_simple_rnn_11/while/add_1:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_11/while/Identity_1IdentityLbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_11/while/Identity_2Identity$backward_simple_rnn_11/while/add:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_11/while/Identity_3IdentityQbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_11/while/Identity_4Identity8backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0"^backward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_11/while/NoOpNoOpG^backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpF^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpH^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0"W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0"[
'backward_simple_rnn_11_while_identity_10backward_simple_rnn_11/while/Identity_1:output:0"[
'backward_simple_rnn_11_while_identity_20backward_simple_rnn_11/while/Identity_2:output:0"[
'backward_simple_rnn_11_while_identity_30backward_simple_rnn_11/while/Identity_3:output:0"[
'backward_simple_rnn_11_while_identity_40backward_simple_rnn_11/while/Identity_4:output:0"¤
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resourceQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resourceRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resourcePbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0"
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpFbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpEbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp2
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpGbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5425085

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
ÿ
ê
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5421008

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
Ú	
Ã
/__inference_sequential_24_layer_call_fn_5422542

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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422080o
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
þ¨
Û
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5422048

inputsY
Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp¢Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp¢backward_simple_rnn_11/while¢?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp¢@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp¢forward_simple_rnn_11/whileQ
forward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:s
)forward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_11/strided_sliceStridedSlice$forward_simple_rnn_11/Shape:output:02forward_simple_rnn_11/strided_slice/stack:output:04forward_simple_rnn_11/strided_slice/stack_1:output:04forward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_11/zeros/packedPack,forward_simple_rnn_11/strided_slice:output:0-forward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_11/zerosFill+forward_simple_rnn_11/zeros/packed:output:0*forward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_11/transpose	Transposeinputs-forward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
forward_simple_rnn_11/Shape_1Shape#forward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_11/strided_slice_1StridedSlice&forward_simple_rnn_11/Shape_1:output:04forward_simple_rnn_11/strided_slice_1/stack:output:06forward_simple_rnn_11/strided_slice_1/stack_1:output:06forward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_11/TensorArrayV2TensorListReserve:forward_simple_rnn_11/TensorArrayV2/element_shape:output:0.forward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¢
=forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_11/transpose:y:0Tforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%forward_simple_rnn_11/strided_slice_2StridedSlice#forward_simple_rnn_11/transpose:y:04forward_simple_rnn_11/strided_slice_2/stack:output:06forward_simple_rnn_11/strided_slice_2/stack_1:output:06forward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_11/simple_rnn_cell_34/MatMulMatMul.forward_simple_rnn_11/strided_slice_2:output:0Fforward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_11/simple_rnn_cell_34/BiasAddBiasAdd9forward_simple_rnn_11/simple_rnn_cell_34/MatMul:product:0Gforward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1MatMul$forward_simple_rnn_11/zeros:output:0Hforward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_11/simple_rnn_cell_34/addAddV29forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd:output:0;forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_11/simple_rnn_cell_34/TanhTanh0forward_simple_rnn_11/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_11/TensorArrayV2_1TensorListReserve<forward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_11/whileWhile1forward_simple_rnn_11/while/loop_counter:output:07forward_simple_rnn_11/while/maximum_iterations:output:0#forward_simple_rnn_11/time:output:0.forward_simple_rnn_11/TensorArrayV2_1:handle:0$forward_simple_rnn_11/zeros:output:0.forward_simple_rnn_11/strided_slice_1:output:0Mforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resourceHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resourceIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_11_while_body_5421871*4
cond,R*
(forward_simple_rnn_11_while_cond_5421870*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_11/while:output:3Oforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_11/strided_slice_3StridedSliceAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_11/strided_slice_3/stack:output:06forward_simple_rnn_11/strided_slice_3/stack_1:output:06forward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_11/transpose_1	TransposeAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
backward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:t
*backward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_11/strided_sliceStridedSlice%backward_simple_rnn_11/Shape:output:03backward_simple_rnn_11/strided_slice/stack:output:05backward_simple_rnn_11/strided_slice/stack_1:output:05backward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_11/zeros/packedPack-backward_simple_rnn_11/strided_slice:output:0.backward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_11/zerosFill,backward_simple_rnn_11/zeros/packed:output:0+backward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
 backward_simple_rnn_11/transpose	Transposeinputs.backward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4r
backward_simple_rnn_11/Shape_1Shape$backward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_11/strided_slice_1StridedSlice'backward_simple_rnn_11/Shape_1:output:05backward_simple_rnn_11/strided_slice_1/stack:output:07backward_simple_rnn_11/strided_slice_1/stack_1:output:07backward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_11/TensorArrayV2TensorListReserve;backward_simple_rnn_11/TensorArrayV2/element_shape:output:0/backward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¹
 backward_simple_rnn_11/ReverseV2	ReverseV2$backward_simple_rnn_11/transpose:y:0.backward_simple_rnn_11/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Lbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ª
>backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_11/ReverseV2:output:0Ubackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
&backward_simple_rnn_11/strided_slice_2StridedSlice$backward_simple_rnn_11/transpose:y:05backward_simple_rnn_11/strided_slice_2/stack:output:07backward_simple_rnn_11/strided_slice_2/stack_1:output:07backward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÈ
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_11/simple_rnn_cell_35/MatMulMatMul/backward_simple_rnn_11/strided_slice_2:output:0Gbackward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_11/simple_rnn_cell_35/BiasAddBiasAdd:backward_simple_rnn_11/simple_rnn_cell_35/MatMul:product:0Hbackward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1MatMul%backward_simple_rnn_11/zeros:output:0Ibackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_11/simple_rnn_cell_35/addAddV2:backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd:output:0<backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_11/simple_rnn_cell_35/TanhTanh1backward_simple_rnn_11/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_11/TensorArrayV2_1TensorListReserve=backward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_11/whileWhile2backward_simple_rnn_11/while/loop_counter:output:08backward_simple_rnn_11/while/maximum_iterations:output:0$backward_simple_rnn_11/time:output:0/backward_simple_rnn_11/TensorArrayV2_1:handle:0%backward_simple_rnn_11/zeros:output:0/backward_simple_rnn_11/strided_slice_1:output:0Nbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resourceIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resourceJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
)backward_simple_rnn_11_while_body_5421979*5
cond-R+
)backward_simple_rnn_11_while_cond_5421978*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_11/while:output:3Pbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_11/strided_slice_3StridedSliceBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_11/strided_slice_3/stack:output:07backward_simple_rnn_11/strided_slice_3/stack_1:output:07backward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_11/transpose_1	TransposeBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_11/strided_slice_3:output:0/backward_simple_rnn_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@^backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpB^backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp^backward_simple_rnn_11/while@^forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?^forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpA^forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp^forward_simple_rnn_11/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp2
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpAbackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp2<
backward_simple_rnn_11/whilebackward_simple_rnn_11/while2
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp2
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp2:
forward_simple_rnn_11/whileforward_simple_rnn_11/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
?
Ë
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5421776

inputsC
1simple_rnn_cell_34_matmul_readvariableop_resource:4@@
2simple_rnn_cell_34_biasadd_readvariableop_resource:@E
3simple_rnn_cell_34_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_34/BiasAdd/ReadVariableOp¢(simple_rnn_cell_34/MatMul/ReadVariableOp¢*simple_rnn_cell_34/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_34/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_34/BiasAddBiasAdd#simple_rnn_cell_34/MatMul:product:01simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_34/MatMul_1MatMulzeros:output:02simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_34/addAddV2#simple_rnn_cell_34/BiasAdd:output:0%simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_34/TanhTanhsimple_rnn_cell_34/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_34_matmul_readvariableop_resource2simple_rnn_cell_34_biasadd_readvariableop_resource3simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
while_body_5421709*
condR
while_cond_5421708*8
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
NoOpNoOp*^simple_rnn_cell_34/BiasAdd/ReadVariableOp)^simple_rnn_cell_34/MatMul/ReadVariableOp+^simple_rnn_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_34/BiasAdd/ReadVariableOp)simple_rnn_cell_34/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_34/MatMul/ReadVariableOp(simple_rnn_cell_34/MatMul/ReadVariableOp2X
*simple_rnn_cell_34/MatMul_1/ReadVariableOp*simple_rnn_cell_34/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ¨
Û
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5422348

inputsY
Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp¢Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp¢backward_simple_rnn_11/while¢?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp¢@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp¢forward_simple_rnn_11/whileQ
forward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:s
)forward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_11/strided_sliceStridedSlice$forward_simple_rnn_11/Shape:output:02forward_simple_rnn_11/strided_slice/stack:output:04forward_simple_rnn_11/strided_slice/stack_1:output:04forward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_11/zeros/packedPack,forward_simple_rnn_11/strided_slice:output:0-forward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_11/zerosFill+forward_simple_rnn_11/zeros/packed:output:0*forward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_simple_rnn_11/transpose	Transposeinputs-forward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4p
forward_simple_rnn_11/Shape_1Shape#forward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_11/strided_slice_1StridedSlice&forward_simple_rnn_11/Shape_1:output:04forward_simple_rnn_11/strided_slice_1/stack:output:06forward_simple_rnn_11/strided_slice_1/stack_1:output:06forward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_11/TensorArrayV2TensorListReserve:forward_simple_rnn_11/TensorArrayV2/element_shape:output:0.forward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¢
=forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_11/transpose:y:0Tforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%forward_simple_rnn_11/strided_slice_2StridedSlice#forward_simple_rnn_11/transpose:y:04forward_simple_rnn_11/strided_slice_2/stack:output:06forward_simple_rnn_11/strided_slice_2/stack_1:output:06forward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÆ
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_11/simple_rnn_cell_34/MatMulMatMul.forward_simple_rnn_11/strided_slice_2:output:0Fforward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_11/simple_rnn_cell_34/BiasAddBiasAdd9forward_simple_rnn_11/simple_rnn_cell_34/MatMul:product:0Gforward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1MatMul$forward_simple_rnn_11/zeros:output:0Hforward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_11/simple_rnn_cell_34/addAddV29forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd:output:0;forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_11/simple_rnn_cell_34/TanhTanh0forward_simple_rnn_11/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_11/TensorArrayV2_1TensorListReserve<forward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_11/whileWhile1forward_simple_rnn_11/while/loop_counter:output:07forward_simple_rnn_11/while/maximum_iterations:output:0#forward_simple_rnn_11/time:output:0.forward_simple_rnn_11/TensorArrayV2_1:handle:0$forward_simple_rnn_11/zeros:output:0.forward_simple_rnn_11/strided_slice_1:output:0Mforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resourceHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resourceIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_11_while_body_5422171*4
cond,R*
(forward_simple_rnn_11_while_cond_5422170*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_11/while:output:3Oforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_11/strided_slice_3StridedSliceAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_11/strided_slice_3/stack:output:06forward_simple_rnn_11/strided_slice_3/stack_1:output:06forward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_11/transpose_1	TransposeAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
backward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:t
*backward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_11/strided_sliceStridedSlice%backward_simple_rnn_11/Shape:output:03backward_simple_rnn_11/strided_slice/stack:output:05backward_simple_rnn_11/strided_slice/stack_1:output:05backward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_11/zeros/packedPack-backward_simple_rnn_11/strided_slice:output:0.backward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_11/zerosFill,backward_simple_rnn_11/zeros/packed:output:0+backward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
 backward_simple_rnn_11/transpose	Transposeinputs.backward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4r
backward_simple_rnn_11/Shape_1Shape$backward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_11/strided_slice_1StridedSlice'backward_simple_rnn_11/Shape_1:output:05backward_simple_rnn_11/strided_slice_1/stack:output:07backward_simple_rnn_11/strided_slice_1/stack_1:output:07backward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_11/TensorArrayV2TensorListReserve;backward_simple_rnn_11/TensorArrayV2/element_shape:output:0/backward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¹
 backward_simple_rnn_11/ReverseV2	ReverseV2$backward_simple_rnn_11/transpose:y:0.backward_simple_rnn_11/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
Lbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ª
>backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_11/ReverseV2:output:0Ubackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
&backward_simple_rnn_11/strided_slice_2StridedSlice$backward_simple_rnn_11/transpose:y:05backward_simple_rnn_11/strided_slice_2/stack:output:07backward_simple_rnn_11/strided_slice_2/stack_1:output:07backward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskÈ
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_11/simple_rnn_cell_35/MatMulMatMul/backward_simple_rnn_11/strided_slice_2:output:0Gbackward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_11/simple_rnn_cell_35/BiasAddBiasAdd:backward_simple_rnn_11/simple_rnn_cell_35/MatMul:product:0Hbackward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1MatMul%backward_simple_rnn_11/zeros:output:0Ibackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_11/simple_rnn_cell_35/addAddV2:backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd:output:0<backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_11/simple_rnn_cell_35/TanhTanh1backward_simple_rnn_11/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_11/TensorArrayV2_1TensorListReserve=backward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_11/whileWhile2backward_simple_rnn_11/while/loop_counter:output:08backward_simple_rnn_11/while/maximum_iterations:output:0$backward_simple_rnn_11/time:output:0/backward_simple_rnn_11/TensorArrayV2_1:handle:0%backward_simple_rnn_11/zeros:output:0/backward_simple_rnn_11/strided_slice_1:output:0Nbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resourceIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resourceJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
)backward_simple_rnn_11_while_body_5422279*5
cond-R+
)backward_simple_rnn_11_while_cond_5422278*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_11/while:output:3Pbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_11/strided_slice_3StridedSliceBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_11/strided_slice_3/stack:output:07backward_simple_rnn_11/strided_slice_3/stack_1:output:07backward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_11/transpose_1	TransposeBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_11/strided_slice_3:output:0/backward_simple_rnn_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@^backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpB^backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp^backward_simple_rnn_11/while@^forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?^forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpA^forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp^forward_simple_rnn_11/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ4: : : : : : 2
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp2
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpAbackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp2<
backward_simple_rnn_11/whilebackward_simple_rnn_11/while2
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp2
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp2:
forward_simple_rnn_11/whileforward_simple_rnn_11/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
¯
Ä
8__inference_backward_simple_rnn_11_layer_call_fn_5424491
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421249o
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
9bidirectional_24_forward_simple_rnn_11_while_body_5422833j
fbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_loop_counterp
lbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_maximum_iterations<
8bidirectional_24_forward_simple_rnn_11_while_placeholder>
:bidirectional_24_forward_simple_rnn_11_while_placeholder_1>
:bidirectional_24_forward_simple_rnn_11_while_placeholder_2i
ebidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_strided_slice_1_0¦
¡bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0r
`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@o
abidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@t
bbidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@9
5bidirectional_24_forward_simple_rnn_11_while_identity;
7bidirectional_24_forward_simple_rnn_11_while_identity_1;
7bidirectional_24_forward_simple_rnn_11_while_identity_2;
7bidirectional_24_forward_simple_rnn_11_while_identity_3;
7bidirectional_24_forward_simple_rnn_11_while_identity_4g
cbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_strided_slice_1¤
bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorp
^bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource:4@m
_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource:@r
`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢Vbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Ubidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp¢Wbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp¯
^bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ê
Pbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¡bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_08bidirectional_24_forward_simple_rnn_11_while_placeholdergbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ö
Ubidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0º
Fbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMulMatMulWbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0]bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
Vbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpabidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¶
Gbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAddBiasAddPbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul:product:0^bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
Wbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpbbidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¡
Hbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1MatMul:bidirectional_24_forward_simple_rnn_11_while_placeholder_2_bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Cbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/addAddV2Pbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd:output:0Rbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
Dbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/TanhTanhGbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Wbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Qbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem:bidirectional_24_forward_simple_rnn_11_while_placeholder_1`bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:0Hbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒt
2bidirectional_24/forward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ñ
0bidirectional_24/forward_simple_rnn_11/while/addAddV28bidirectional_24_forward_simple_rnn_11_while_placeholder;bidirectional_24/forward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: v
4bidirectional_24/forward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
2bidirectional_24/forward_simple_rnn_11/while/add_1AddV2fbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_loop_counter=bidirectional_24/forward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: Î
5bidirectional_24/forward_simple_rnn_11/while/IdentityIdentity6bidirectional_24/forward_simple_rnn_11/while/add_1:z:02^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
7bidirectional_24/forward_simple_rnn_11/while/Identity_1Identitylbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_maximum_iterations2^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Î
7bidirectional_24/forward_simple_rnn_11/while/Identity_2Identity4bidirectional_24/forward_simple_rnn_11/while/add:z:02^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: û
7bidirectional_24/forward_simple_rnn_11/while/Identity_3Identityabidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ó
7bidirectional_24/forward_simple_rnn_11/while/Identity_4IdentityHbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:02^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@þ
1bidirectional_24/forward_simple_rnn_11/while/NoOpNoOpW^bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpV^bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpX^bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ì
cbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_strided_slice_1ebidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_strided_slice_1_0"w
5bidirectional_24_forward_simple_rnn_11_while_identity>bidirectional_24/forward_simple_rnn_11/while/Identity:output:0"{
7bidirectional_24_forward_simple_rnn_11_while_identity_1@bidirectional_24/forward_simple_rnn_11/while/Identity_1:output:0"{
7bidirectional_24_forward_simple_rnn_11_while_identity_2@bidirectional_24/forward_simple_rnn_11/while/Identity_2:output:0"{
7bidirectional_24_forward_simple_rnn_11_while_identity_3@bidirectional_24/forward_simple_rnn_11/while/Identity_3:output:0"{
7bidirectional_24_forward_simple_rnn_11_while_identity_4@bidirectional_24/forward_simple_rnn_11/while/Identity_4:output:0"Ä
_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resourceabidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"Æ
`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resourcebbidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0"Â
^bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0"Æ
bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor¡bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2°
Vbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpVbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2®
Ubidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpUbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp2²
Wbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpWbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
while_body_5424894
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_35_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_35/MatMul/ReadVariableOp¢0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_35/BiasAddBiasAdd)while/simple_rnn_cell_35/MatMul:product:07while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_35/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_35/addAddV2)while/simple_rnn_cell_35/BiasAdd:output:0+while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_35/TanhTanh while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_35/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_35/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_35/MatMul/ReadVariableOp1^while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_35_matmul_readvariableop_resource9while_simple_rnn_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_35/MatMul/ReadVariableOp.while/simple_rnn_cell_35/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
while_body_5424402
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_34_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_34/MatMul/ReadVariableOp¢0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_34/BiasAddBiasAdd)while/simple_rnn_cell_34/MatMul:product:07while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_34/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_34/addAddV2)while/simple_rnn_cell_34/BiasAdd:output:0+while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_34/TanhTanh while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_34/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_34/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_34/MatMul/ReadVariableOp1^while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_34_matmul_readvariableop_resource9while_simple_rnn_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_34/MatMul/ReadVariableOp.while/simple_rnn_cell_34/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
ÉS

:bidirectional_24_backward_simple_rnn_11_while_body_5422941l
hbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_loop_counterr
nbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_maximum_iterations=
9bidirectional_24_backward_simple_rnn_11_while_placeholder?
;bidirectional_24_backward_simple_rnn_11_while_placeholder_1?
;bidirectional_24_backward_simple_rnn_11_while_placeholder_2k
gbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_strided_slice_1_0¨
£bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0s
abidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@p
bbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@u
cbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@:
6bidirectional_24_backward_simple_rnn_11_while_identity<
8bidirectional_24_backward_simple_rnn_11_while_identity_1<
8bidirectional_24_backward_simple_rnn_11_while_identity_2<
8bidirectional_24_backward_simple_rnn_11_while_identity_3<
8bidirectional_24_backward_simple_rnn_11_while_identity_4i
ebidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_strided_slice_1¦
¡bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorq
_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource:4@n
`bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource:@s
abidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢Wbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Vbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp¢Xbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp°
_bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ï
Qbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem£bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_09bidirectional_24_backward_simple_rnn_11_while_placeholderhbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ø
Vbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpabidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0½
Gbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMulMatMulXbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0^bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ö
Wbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpbbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¹
Hbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAddBiasAddQbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul:product:0_bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ü
Xbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpcbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¤
Ibidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1MatMul;bidirectional_24_backward_simple_rnn_11_while_placeholder_2`bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
Dbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/addAddV2Qbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd:output:0Sbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@É
Ebidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/TanhTanhHbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Xbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Rbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem;bidirectional_24_backward_simple_rnn_11_while_placeholder_1abidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:0Ibidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒu
3bidirectional_24/backward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ô
1bidirectional_24/backward_simple_rnn_11/while/addAddV29bidirectional_24_backward_simple_rnn_11_while_placeholder<bidirectional_24/backward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: w
5bidirectional_24/backward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
3bidirectional_24/backward_simple_rnn_11/while/add_1AddV2hbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_loop_counter>bidirectional_24/backward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: Ñ
6bidirectional_24/backward_simple_rnn_11/while/IdentityIdentity7bidirectional_24/backward_simple_rnn_11/while/add_1:z:03^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
8bidirectional_24/backward_simple_rnn_11/while/Identity_1Identitynbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_maximum_iterations3^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ñ
8bidirectional_24/backward_simple_rnn_11/while/Identity_2Identity5bidirectional_24/backward_simple_rnn_11/while/add:z:03^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: þ
8bidirectional_24/backward_simple_rnn_11/while/Identity_3Identitybbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ö
8bidirectional_24/backward_simple_rnn_11/while/Identity_4IdentityIbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:03^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2bidirectional_24/backward_simple_rnn_11/while/NoOpNoOpX^bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpW^bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpY^bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ð
ebidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_strided_slice_1gbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_strided_slice_1_0"y
6bidirectional_24_backward_simple_rnn_11_while_identity?bidirectional_24/backward_simple_rnn_11/while/Identity:output:0"}
8bidirectional_24_backward_simple_rnn_11_while_identity_1Abidirectional_24/backward_simple_rnn_11/while/Identity_1:output:0"}
8bidirectional_24_backward_simple_rnn_11_while_identity_2Abidirectional_24/backward_simple_rnn_11/while/Identity_2:output:0"}
8bidirectional_24_backward_simple_rnn_11_while_identity_3Abidirectional_24/backward_simple_rnn_11/while/Identity_3:output:0"}
8bidirectional_24_backward_simple_rnn_11_while_identity_4Abidirectional_24/backward_simple_rnn_11/while/Identity_4:output:0"Æ
`bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resourcebbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"È
abidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resourcecbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"Ä
_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resourceabidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0"Ê
¡bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor£bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2²
Wbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpWbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2°
Vbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpVbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp2´
Xbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpXbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
while_cond_5421021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5421021___redundant_placeholder05
1while_while_cond_5421021___redundant_placeholder15
1while_while_cond_5421021___redundant_placeholder25
1while_while_cond_5421021___redundant_placeholder3
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

î
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5421504

inputs/
forward_simple_rnn_11_5421375:4@+
forward_simple_rnn_11_5421377:@/
forward_simple_rnn_11_5421379:@@0
backward_simple_rnn_11_5421494:4@,
backward_simple_rnn_11_5421496:@0
backward_simple_rnn_11_5421498:@@
identity¢.backward_simple_rnn_11/StatefulPartitionedCall¢-forward_simple_rnn_11/StatefulPartitionedCallË
-forward_simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_11_5421375forward_simple_rnn_11_5421377forward_simple_rnn_11_5421379*
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5421374Ð
.backward_simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_11_5421494backward_simple_rnn_11_5421496backward_simple_rnn_11_5421498*
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421493M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Õ
concatConcatV26forward_simple_rnn_11/StatefulPartitionedCall:output:07backward_simple_rnn_11/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp/^backward_simple_rnn_11/StatefulPartitionedCall.^forward_simple_rnn_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2`
.backward_simple_rnn_11/StatefulPartitionedCall.backward_simple_rnn_11/StatefulPartitionedCall2^
-forward_simple_rnn_11/StatefulPartitionedCall-forward_simple_rnn_11/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
¯
while_cond_5424781
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5424781___redundant_placeholder05
1while_while_cond_5424781___redundant_placeholder15
1while_while_cond_5424781___redundant_placeholder25
1while_while_cond_5424781___redundant_placeholder3
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
while_body_5424782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_35_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_35/MatMul/ReadVariableOp¢0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_35/BiasAddBiasAdd)while/simple_rnn_cell_35/MatMul:product:07while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_35/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_35/addAddV2)while/simple_rnn_cell_35/BiasAdd:output:0+while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_35/TanhTanh while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_35/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_35/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_35/MatMul/ReadVariableOp1^while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_35_matmul_readvariableop_resource9while_simple_rnn_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_35/MatMul/ReadVariableOp.while/simple_rnn_cell_35/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
Hsequential_24_bidirectional_24_backward_simple_rnn_11_while_cond_5420585
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_loop_counter
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_maximum_iterationsK
Gsequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholderM
Isequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholder_1M
Isequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholder_2
sequential_24_bidirectional_24_backward_simple_rnn_11_while_less_sequential_24_bidirectional_24_backward_simple_rnn_11_strided_slice_1¢
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_cond_5420585___redundant_placeholder0¢
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_cond_5420585___redundant_placeholder1¢
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_cond_5420585___redundant_placeholder2¢
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_cond_5420585___redundant_placeholder3H
Dsequential_24_bidirectional_24_backward_simple_rnn_11_while_identity
»
@sequential_24/bidirectional_24/backward_simple_rnn_11/while/LessLessGsequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholdersequential_24_bidirectional_24_backward_simple_rnn_11_while_less_sequential_24_bidirectional_24_backward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: ·
Dsequential_24/bidirectional_24/backward_simple_rnn_11/while/IdentityIdentityDsequential_24/bidirectional_24/backward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "
Dsequential_24_bidirectional_24_backward_simple_rnn_11_while_identityMsequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity:output:0*(
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
Gsequential_24_bidirectional_24_forward_simple_rnn_11_while_cond_5420477
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_loop_counter
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_maximum_iterationsJ
Fsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholderL
Hsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholder_1L
Hsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholder_2
sequential_24_bidirectional_24_forward_simple_rnn_11_while_less_sequential_24_bidirectional_24_forward_simple_rnn_11_strided_slice_1 
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_cond_5420477___redundant_placeholder0 
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_cond_5420477___redundant_placeholder1 
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_cond_5420477___redundant_placeholder2 
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_cond_5420477___redundant_placeholder3G
Csequential_24_bidirectional_24_forward_simple_rnn_11_while_identity
·
?sequential_24/bidirectional_24/forward_simple_rnn_11/while/LessLessFsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholdersequential_24_bidirectional_24_forward_simple_rnn_11_while_less_sequential_24_bidirectional_24_forward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: µ
Csequential_24/bidirectional_24/forward_simple_rnn_11/while/IdentityIdentityCsequential_24/bidirectional_24/forward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "
Csequential_24_bidirectional_24_forward_simple_rnn_11_while_identityLsequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity:output:0*(
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
¹
Á
7__inference_forward_simple_rnn_11_layer_call_fn_5424018

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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5421374o
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
²
Ñ
(forward_simple_rnn_11_while_cond_5423567H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2J
Fforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423567___redundant_placeholder0a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423567___redundant_placeholder1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423567___redundant_placeholder2a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423567___redundant_placeholder3(
$forward_simple_rnn_11_while_identity
º
 forward_simple_rnn_11/while/LessLess'forward_simple_rnn_11_while_placeholderFforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_11/while/IdentityIdentity$forward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0*(
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
E__inference_dense_24_layer_call_and_return_conditional_losses_5422073

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
û>
Í
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424249
inputs_0C
1simple_rnn_cell_34_matmul_readvariableop_resource:4@@
2simple_rnn_cell_34_biasadd_readvariableop_resource:@E
3simple_rnn_cell_34_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_34/BiasAdd/ReadVariableOp¢(simple_rnn_cell_34/MatMul/ReadVariableOp¢*simple_rnn_cell_34/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_34/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_34/BiasAddBiasAdd#simple_rnn_cell_34/MatMul:product:01simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_34/MatMul_1MatMulzeros:output:02simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_34/addAddV2#simple_rnn_cell_34/BiasAdd:output:0%simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_34/TanhTanhsimple_rnn_cell_34/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_34_matmul_readvariableop_resource2simple_rnn_cell_34_biasadd_readvariableop_resource3simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
while_body_5424182*
condR
while_cond_5424181*8
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
NoOpNoOp*^simple_rnn_cell_34/BiasAdd/ReadVariableOp)^simple_rnn_cell_34/MatMul/ReadVariableOp+^simple_rnn_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_34/BiasAdd/ReadVariableOp)simple_rnn_cell_34/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_34/MatMul/ReadVariableOp(simple_rnn_cell_34/MatMul/ReadVariableOp2X
*simple_rnn_cell_34/MatMul_1/ReadVariableOp*simple_rnn_cell_34/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0

	
9bidirectional_24_forward_simple_rnn_11_while_cond_5422832j
fbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_loop_counterp
lbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_maximum_iterations<
8bidirectional_24_forward_simple_rnn_11_while_placeholder>
:bidirectional_24_forward_simple_rnn_11_while_placeholder_1>
:bidirectional_24_forward_simple_rnn_11_while_placeholder_2l
hbidirectional_24_forward_simple_rnn_11_while_less_bidirectional_24_forward_simple_rnn_11_strided_slice_1
bidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_cond_5422832___redundant_placeholder0
bidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_cond_5422832___redundant_placeholder1
bidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_cond_5422832___redundant_placeholder2
bidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_cond_5422832___redundant_placeholder39
5bidirectional_24_forward_simple_rnn_11_while_identity
þ
1bidirectional_24/forward_simple_rnn_11/while/LessLess8bidirectional_24_forward_simple_rnn_11_while_placeholderhbidirectional_24_forward_simple_rnn_11_while_less_bidirectional_24_forward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: 
5bidirectional_24/forward_simple_rnn_11/while/IdentityIdentity5bidirectional_24/forward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "w
5bidirectional_24_forward_simple_rnn_11_while_identity>bidirectional_24/forward_simple_rnn_11/while/Identity:output:0*(
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
while_cond_5424893
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5424893___redundant_placeholder05
1while_while_cond_5424893___redundant_placeholder15
1while_while_cond_5424893___redundant_placeholder25
1while_while_cond_5424893___redundant_placeholder3
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
Í
ä
)backward_simple_rnn_11_while_cond_5421978J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2L
Hbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5421978___redundant_placeholder0c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5421978___redundant_placeholder1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5421978___redundant_placeholder2c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5421978___redundant_placeholder3)
%backward_simple_rnn_11_while_identity
¾
!backward_simple_rnn_11/while/LessLess(backward_simple_rnn_11_while_placeholderHbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_11/while/IdentityIdentity%backward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0*(
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
4__inference_simple_rnn_cell_35_layer_call_fn_5425037

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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5421008o
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
while_body_5421577
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_35_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_35/MatMul/ReadVariableOp¢0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_35/BiasAddBiasAdd)while/simple_rnn_cell_35/MatMul:product:07while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_35/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_35/addAddV2)while/simple_rnn_cell_35/BiasAdd:output:0+while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_35/TanhTanh while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_35/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_35/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_35/MatMul/ReadVariableOp1^while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_35_matmul_readvariableop_resource9while_simple_rnn_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_35/MatMul/ReadVariableOp.while/simple_rnn_cell_35/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
)backward_simple_rnn_11_while_cond_5423235J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2L
Hbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423235___redundant_placeholder0c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423235___redundant_placeholder1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423235___redundant_placeholder2c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423235___redundant_placeholder3)
%backward_simple_rnn_11_while_identity
¾
!backward_simple_rnn_11/while/LessLess(backward_simple_rnn_11_while_placeholderHbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_11/while/IdentityIdentity%backward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0*(
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
while_body_5424558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_35_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_35/MatMul/ReadVariableOp¢0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_35/BiasAddBiasAdd)while/simple_rnn_cell_35/MatMul:product:07while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_35/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_35/addAddV2)while/simple_rnn_cell_35/BiasAdd:output:0+while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_35/TanhTanh while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_35/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_35/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_35/MatMul/ReadVariableOp1^while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_35_matmul_readvariableop_resource9while_simple_rnn_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_35/MatMul/ReadVariableOp.while/simple_rnn_cell_35/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424359

inputsC
1simple_rnn_cell_34_matmul_readvariableop_resource:4@@
2simple_rnn_cell_34_biasadd_readvariableop_resource:@E
3simple_rnn_cell_34_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_34/BiasAdd/ReadVariableOp¢(simple_rnn_cell_34/MatMul/ReadVariableOp¢*simple_rnn_cell_34/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_34/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_34/BiasAddBiasAdd#simple_rnn_cell_34/MatMul:product:01simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_34/MatMul_1MatMulzeros:output:02simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_34/addAddV2#simple_rnn_cell_34/BiasAdd:output:0%simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_34/TanhTanhsimple_rnn_cell_34/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_34_matmul_readvariableop_resource2simple_rnn_cell_34_biasadd_readvariableop_resource3simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
while_body_5424292*
condR
while_cond_5424291*8
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
NoOpNoOp*^simple_rnn_cell_34/BiasAdd/ReadVariableOp)^simple_rnn_cell_34/MatMul/ReadVariableOp+^simple_rnn_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_34/BiasAdd/ReadVariableOp)simple_rnn_cell_34/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_34/MatMul/ReadVariableOp(simple_rnn_cell_34/MatMul/ReadVariableOp2X
*simple_rnn_cell_34/MatMul_1/ReadVariableOp*simple_rnn_cell_34/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ì
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5425006

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
ÝB
è
(forward_simple_rnn_11_while_body_5423128H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2G
Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0
forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_11_while_identity*
&forward_simple_rnn_11_while_identity_1*
&forward_simple_rnn_11_while_identity_2*
&forward_simple_rnn_11_while_identity_3*
&forward_simple_rnn_11_while_identity_4E
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp¢Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
Mforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
?forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_11_while_placeholderVforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ô
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_11/while/simple_rnn_cell_34/MatMulMatMulFforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAddBiasAdd?forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul:product:0Mforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1MatMul)forward_simple_rnn_11_while_placeholder_2Nforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_11/while/simple_rnn_cell_34/addAddV2?forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd:output:0Aforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_11/while/simple_rnn_cell_34/TanhTanh6forward_simple_rnn_11/while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_11_while_placeholder_1Oforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_11/while/addAddV2'forward_simple_rnn_11_while_placeholder*forward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_11/while/add_1AddV2Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counter,forward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_11/while/IdentityIdentity%forward_simple_rnn_11/while/add_1:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_11/while/Identity_1IdentityJforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_11/while/Identity_2Identity#forward_simple_rnn_11/while/add:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_11/while/Identity_3IdentityPforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_11/while/Identity_4Identity7forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0!^forward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_11/while/NoOpNoOpF^forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpE^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpG^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0"U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0"Y
&forward_simple_rnn_11_while_identity_1/forward_simple_rnn_11/while/Identity_1:output:0"Y
&forward_simple_rnn_11_while_identity_2/forward_simple_rnn_11/while/Identity_2:output:0"Y
&forward_simple_rnn_11_while_identity_3/forward_simple_rnn_11/while/Identity_3:output:0"Y
&forward_simple_rnn_11_while_identity_4/forward_simple_rnn_11/while/Identity_4:output:0"¢
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resourcePforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resourceQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resourceOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0"
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpEforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpDforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp2
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpFforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
´
ü
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422080

inputs*
bidirectional_24_5422049:4@&
bidirectional_24_5422051:@*
bidirectional_24_5422053:@@*
bidirectional_24_5422055:4@&
bidirectional_24_5422057:@*
bidirectional_24_5422059:@@#
dense_24_5422074:	
dense_24_5422076:
identity¢(bidirectional_24/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall
(bidirectional_24/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_24_5422049bidirectional_24_5422051bidirectional_24_5422053bidirectional_24_5422055bidirectional_24_5422057bidirectional_24_5422059*
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5422048¡
 dense_24/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_24/StatefulPartitionedCall:output:0dense_24_5422074dense_24_5422076*
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
E__inference_dense_24_layer_call_and_return_conditional_losses_5422073x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_24/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_24/StatefulPartitionedCall(bidirectional_24/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
²
Ñ
(forward_simple_rnn_11_while_cond_5421870H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2J
Fforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5421870___redundant_placeholder0a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5421870___redundant_placeholder1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5421870___redundant_placeholder2a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5421870___redundant_placeholder3(
$forward_simple_rnn_11_while_identity
º
 forward_simple_rnn_11/while/LessLess'forward_simple_rnn_11_while_placeholderFforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_11/while/IdentityIdentity$forward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0*(
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5420949

inputs,
simple_rnn_cell_34_5420872:4@(
simple_rnn_cell_34_5420874:@,
simple_rnn_cell_34_5420876:@@
identity¢*simple_rnn_cell_34/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_34_5420872simple_rnn_cell_34_5420874simple_rnn_cell_34_5420876*
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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5420832n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_34_5420872simple_rnn_cell_34_5420874simple_rnn_cell_34_5420876*
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
while_body_5420885*
condR
while_cond_5420884*8
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
NoOpNoOp+^simple_rnn_cell_34/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_34/StatefulPartitionedCall*simple_rnn_cell_34/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ß
¯
while_cond_5421306
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5421306___redundant_placeholder05
1while_while_cond_5421306___redundant_placeholder15
1while_while_cond_5421306___redundant_placeholder25
1while_while_cond_5421306___redundant_placeholder3
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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422470
bidirectional_24_input*
bidirectional_24_5422451:4@&
bidirectional_24_5422453:@*
bidirectional_24_5422455:@@*
bidirectional_24_5422457:4@&
bidirectional_24_5422459:@*
bidirectional_24_5422461:@@#
dense_24_5422464:	
dense_24_5422466:
identity¢(bidirectional_24/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall
(bidirectional_24/StatefulPartitionedCallStatefulPartitionedCallbidirectional_24_inputbidirectional_24_5422451bidirectional_24_5422453bidirectional_24_5422455bidirectional_24_5422457bidirectional_24_5422459bidirectional_24_5422461*
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5422048¡
 dense_24/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_24/StatefulPartitionedCall:output:0dense_24_5422464dense_24_5422466*
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
E__inference_dense_24_layer_call_and_return_conditional_losses_5422073x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_24/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_24/StatefulPartitionedCall(bidirectional_24/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_24_input
½"
ß
while_body_5421022
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_35_5421044_0:4@0
"while_simple_rnn_cell_35_5421046_0:@4
"while_simple_rnn_cell_35_5421048_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_35_5421044:4@.
 while_simple_rnn_cell_35_5421046:@2
 while_simple_rnn_cell_35_5421048:@@¢0while/simple_rnn_cell_35/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_35/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_35_5421044_0"while_simple_rnn_cell_35_5421046_0"while_simple_rnn_cell_35_5421048_0*
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5421008r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_35/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_35/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_35_5421044"while_simple_rnn_cell_35_5421044_0"F
 while_simple_rnn_cell_35_5421046"while_simple_rnn_cell_35_5421046_0"F
 while_simple_rnn_cell_35_5421048"while_simple_rnn_cell_35_5421048_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_35/StatefulPartitionedCall0while/simple_rnn_cell_35/StatefulPartitionedCall: 
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424625
inputs_0C
1simple_rnn_cell_35_matmul_readvariableop_resource:4@@
2simple_rnn_cell_35_biasadd_readvariableop_resource:@E
3simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_35/BiasAdd/ReadVariableOp¢(simple_rnn_cell_35/MatMul/ReadVariableOp¢*simple_rnn_cell_35/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_35/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_35/BiasAddBiasAdd#simple_rnn_cell_35/MatMul:product:01simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_35/MatMul_1MatMulzeros:output:02simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_35/addAddV2#simple_rnn_cell_35/BiasAdd:output:0%simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_35/TanhTanhsimple_rnn_cell_35/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_35_matmul_readvariableop_resource2simple_rnn_cell_35_biasadd_readvariableop_resource3simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
while_body_5424558*
condR
while_cond_5424557*8
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
NoOpNoOp*^simple_rnn_cell_35/BiasAdd/ReadVariableOp)^simple_rnn_cell_35/MatMul/ReadVariableOp+^simple_rnn_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_35/BiasAdd/ReadVariableOp)simple_rnn_cell_35/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_35/MatMul/ReadVariableOp(simple_rnn_cell_35/MatMul/ReadVariableOp2X
*simple_rnn_cell_35/MatMul_1/ReadVariableOp*simple_rnn_cell_35/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ß
¯
while_cond_5424557
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5424557___redundant_placeholder05
1while_while_cond_5424557___redundant_placeholder15
1while_while_cond_5424557___redundant_placeholder25
1while_while_cond_5424557___redundant_placeholder3
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
while_cond_5421425
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5421425___redundant_placeholder05
1while_while_cond_5421425___redundant_placeholder15
1while_while_cond_5421425___redundant_placeholder25
1while_while_cond_5421425___redundant_placeholder3
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
A
Ì
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421644

inputsC
1simple_rnn_cell_35_matmul_readvariableop_resource:4@@
2simple_rnn_cell_35_biasadd_readvariableop_resource:@E
3simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_35/BiasAdd/ReadVariableOp¢(simple_rnn_cell_35/MatMul/ReadVariableOp¢*simple_rnn_cell_35/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_35/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_35/BiasAddBiasAdd#simple_rnn_cell_35/MatMul:product:01simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_35/MatMul_1MatMulzeros:output:02simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_35/addAddV2#simple_rnn_cell_35/BiasAdd:output:0%simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_35/TanhTanhsimple_rnn_cell_35/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_35_matmul_readvariableop_resource2simple_rnn_cell_35_biasadd_readvariableop_resource3simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
while_body_5421577*
condR
while_cond_5421576*8
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
NoOpNoOp*^simple_rnn_cell_35/BiasAdd/ReadVariableOp)^simple_rnn_cell_35/MatMul/ReadVariableOp+^simple_rnn_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_35/BiasAdd/ReadVariableOp)simple_rnn_cell_35/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_35/MatMul/ReadVariableOp(simple_rnn_cell_35/MatMul/ReadVariableOp2X
*simple_rnn_cell_35/MatMul_1/ReadVariableOp*simple_rnn_cell_35/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
Ñ
(forward_simple_rnn_11_while_cond_5423787H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2J
Fforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423787___redundant_placeholder0a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423787___redundant_placeholder1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423787___redundant_placeholder2a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423787___redundant_placeholder3(
$forward_simple_rnn_11_while_identity
º
 forward_simple_rnn_11/while/LessLess'forward_simple_rnn_11_while_placeholderFforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_11/while/IdentityIdentity$forward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0*(
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
2__inference_bidirectional_24_layer_call_fn_5423034
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5421504p
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
ÿ6
­
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421086

inputs,
simple_rnn_cell_35_5421009:4@(
simple_rnn_cell_35_5421011:@,
simple_rnn_cell_35_5421013:@@
identity¢*simple_rnn_cell_35/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_35_5421009simple_rnn_cell_35_5421011simple_rnn_cell_35_5421013*
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5421008n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_35_5421009simple_rnn_cell_35_5421011simple_rnn_cell_35_5421013*
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
while_body_5421022*
condR
while_cond_5421021*8
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
NoOpNoOp+^simple_rnn_cell_35/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_35/StatefulPartitionedCall*simple_rnn_cell_35/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ï`
µ
Hsequential_24_bidirectional_24_backward_simple_rnn_11_while_body_5420586
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_loop_counter
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_maximum_iterationsK
Gsequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholderM
Isequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholder_1M
Isequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholder_2
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_strided_slice_1_0Ä
¿sequential_24_bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0
osequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@~
psequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@
qsequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@H
Dsequential_24_bidirectional_24_backward_simple_rnn_11_while_identityJ
Fsequential_24_bidirectional_24_backward_simple_rnn_11_while_identity_1J
Fsequential_24_bidirectional_24_backward_simple_rnn_11_while_identity_2J
Fsequential_24_bidirectional_24_backward_simple_rnn_11_while_identity_3J
Fsequential_24_bidirectional_24_backward_simple_rnn_11_while_identity_4
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_strided_slice_1Â
½sequential_24_bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor
msequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource:4@|
nsequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource:@
osequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢esequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢dsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp¢fsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp¾
msequential_24/bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   µ
_sequential_24/bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¿sequential_24_bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0Gsequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholdervsequential_24/bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0
dsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOposequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0ç
Usequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMulMatMulfsequential_24/bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0lsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
esequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOppsequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0ã
Vsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAddBiasAdd_sequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul:product:0msequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
fsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpqsequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Î
Wsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1MatMulIsequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholder_2nsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ñ
Rsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/addAddV2_sequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd:output:0asequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@å
Ssequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/TanhTanhVsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
fsequential_24/bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
`sequential_24/bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemIsequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholder_1osequential_24/bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:0Wsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ
Asequential_24/bidirectional_24/backward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :þ
?sequential_24/bidirectional_24/backward_simple_rnn_11/while/addAddV2Gsequential_24_bidirectional_24_backward_simple_rnn_11_while_placeholderJsequential_24/bidirectional_24/backward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: 
Csequential_24/bidirectional_24/backward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :À
Asequential_24/bidirectional_24/backward_simple_rnn_11/while/add_1AddV2sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_loop_counterLsequential_24/bidirectional_24/backward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: û
Dsequential_24/bidirectional_24/backward_simple_rnn_11/while/IdentityIdentityEsequential_24/bidirectional_24/backward_simple_rnn_11/while/add_1:z:0A^sequential_24/bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ã
Fsequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity_1Identitysequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_while_maximum_iterationsA^sequential_24/bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: û
Fsequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity_2IdentityCsequential_24/bidirectional_24/backward_simple_rnn_11/while/add:z:0A^sequential_24/bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ¨
Fsequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity_3Identitypsequential_24/bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0A^sequential_24/bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
:  
Fsequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity_4IdentityWsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0A^sequential_24/bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
@sequential_24/bidirectional_24/backward_simple_rnn_11/while/NoOpNoOpf^sequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpe^sequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpg^sequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Dsequential_24_bidirectional_24_backward_simple_rnn_11_while_identityMsequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity:output:0"
Fsequential_24_bidirectional_24_backward_simple_rnn_11_while_identity_1Osequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity_1:output:0"
Fsequential_24_bidirectional_24_backward_simple_rnn_11_while_identity_2Osequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity_2:output:0"
Fsequential_24_bidirectional_24_backward_simple_rnn_11_while_identity_3Osequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity_3:output:0"
Fsequential_24_bidirectional_24_backward_simple_rnn_11_while_identity_4Osequential_24/bidirectional_24/backward_simple_rnn_11/while/Identity_4:output:0"
sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_strided_slice_1sequential_24_bidirectional_24_backward_simple_rnn_11_while_sequential_24_bidirectional_24_backward_simple_rnn_11_strided_slice_1_0"â
nsequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resourcepsequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"ä
osequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resourceqsequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"à
msequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resourceosequential_24_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0"
½sequential_24_bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor¿sequential_24_bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2Î
esequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpesequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2Ì
dsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpdsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp2Ð
fsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpfsequential_24/bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
ûñ
á
"__inference__wrapped_model_5420662
bidirectional_24_inputx
fsequential_24_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource:4@u
gsequential_24_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource:@z
hsequential_24_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@y
gsequential_24_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource:4@v
hsequential_24_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource:@{
isequential_24_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@H
5sequential_24_dense_24_matmul_readvariableop_resource:	D
6sequential_24_dense_24_biasadd_readvariableop_resource:
identity¢_sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢^sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp¢`sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp¢;sequential_24/bidirectional_24/backward_simple_rnn_11/while¢^sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢]sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp¢_sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp¢:sequential_24/bidirectional_24/forward_simple_rnn_11/while¢-sequential_24/dense_24/BiasAdd/ReadVariableOp¢,sequential_24/dense_24/MatMul/ReadVariableOp
:sequential_24/bidirectional_24/forward_simple_rnn_11/ShapeShapebidirectional_24_input*
T0*
_output_shapes
:
Hsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ú
Bsequential_24/bidirectional_24/forward_simple_rnn_11/strided_sliceStridedSliceCsequential_24/bidirectional_24/forward_simple_rnn_11/Shape:output:0Qsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice/stack:output:0Ssequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice/stack_1:output:0Ssequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Csequential_24/bidirectional_24/forward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
Asequential_24/bidirectional_24/forward_simple_rnn_11/zeros/packedPackKsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice:output:0Lsequential_24/bidirectional_24/forward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
@sequential_24/bidirectional_24/forward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
:sequential_24/bidirectional_24/forward_simple_rnn_11/zerosFillJsequential_24/bidirectional_24/forward_simple_rnn_11/zeros/packed:output:0Isequential_24/bidirectional_24/forward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Csequential_24/bidirectional_24/forward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ç
>sequential_24/bidirectional_24/forward_simple_rnn_11/transpose	Transposebidirectional_24_inputLsequential_24/bidirectional_24/forward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4®
<sequential_24/bidirectional_24/forward_simple_rnn_11/Shape_1ShapeBsequential_24/bidirectional_24/forward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:
Jsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ä
Dsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_1StridedSliceEsequential_24/bidirectional_24/forward_simple_rnn_11/Shape_1:output:0Ssequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_1/stack:output:0Usequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_1:output:0Usequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Psequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
Bsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2TensorListReserveYsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2/element_shape:output:0Msequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ»
jsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ÿ
\sequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorBsequential_24/bidirectional_24/forward_simple_rnn_11/transpose:y:0ssequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Dsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_2StridedSliceBsequential_24/bidirectional_24/forward_simple_rnn_11/transpose:y:0Ssequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_2/stack:output:0Usequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_1:output:0Usequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
]sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpfsequential_24_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0À
Nsequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMulMatMulMsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_2:output:0esequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
^sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpgsequential_24_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Î
Osequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAddBiasAddXsequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul:product:0fsequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
_sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOphsequential_24_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0º
Psequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1MatMulCsequential_24/bidirectional_24/forward_simple_rnn_11/zeros:output:0gsequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
Ksequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/addAddV2Xsequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd:output:0Zsequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@×
Lsequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/TanhTanhOsequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
Rsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Qsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ä
Dsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1TensorListReserve[sequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0Zsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ{
9sequential_24/bidirectional_24/forward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Msequential_24/bidirectional_24/forward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Gsequential_24/bidirectional_24/forward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
:sequential_24/bidirectional_24/forward_simple_rnn_11/whileWhilePsequential_24/bidirectional_24/forward_simple_rnn_11/while/loop_counter:output:0Vsequential_24/bidirectional_24/forward_simple_rnn_11/while/maximum_iterations:output:0Bsequential_24/bidirectional_24/forward_simple_rnn_11/time:output:0Msequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1:handle:0Csequential_24/bidirectional_24/forward_simple_rnn_11/zeros:output:0Msequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_1:output:0lsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0fsequential_24_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resourcegsequential_24_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resourcehsequential_24_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
Gsequential_24_bidirectional_24_forward_simple_rnn_11_while_body_5420478*S
condKRI
Gsequential_24_bidirectional_24_forward_simple_rnn_11_while_cond_5420477*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¶
esequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   õ
Wsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStackCsequential_24/bidirectional_24/forward_simple_rnn_11/while:output:3nsequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
Jsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Lsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_3StridedSlice`sequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Ssequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_3/stack:output:0Usequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_1:output:0Usequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
Esequential_24/bidirectional_24/forward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
@sequential_24/bidirectional_24/forward_simple_rnn_11/transpose_1	Transpose`sequential_24/bidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Nsequential_24/bidirectional_24/forward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
;sequential_24/bidirectional_24/backward_simple_rnn_11/ShapeShapebidirectional_24_input*
T0*
_output_shapes
:
Isequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ß
Csequential_24/bidirectional_24/backward_simple_rnn_11/strided_sliceStridedSliceDsequential_24/bidirectional_24/backward_simple_rnn_11/Shape:output:0Rsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice/stack:output:0Tsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice/stack_1:output:0Tsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Dsequential_24/bidirectional_24/backward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
Bsequential_24/bidirectional_24/backward_simple_rnn_11/zeros/packedPackLsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice:output:0Msequential_24/bidirectional_24/backward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
Asequential_24/bidirectional_24/backward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
;sequential_24/bidirectional_24/backward_simple_rnn_11/zerosFillKsequential_24/bidirectional_24/backward_simple_rnn_11/zeros/packed:output:0Jsequential_24/bidirectional_24/backward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dsequential_24/bidirectional_24/backward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          é
?sequential_24/bidirectional_24/backward_simple_rnn_11/transpose	Transposebidirectional_24_inputMsequential_24/bidirectional_24/backward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4°
=sequential_24/bidirectional_24/backward_simple_rnn_11/Shape_1ShapeCsequential_24/bidirectional_24/backward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:
Ksequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Msequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Msequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
Esequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_1StridedSliceFsequential_24/bidirectional_24/backward_simple_rnn_11/Shape_1:output:0Tsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_1/stack:output:0Vsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_1:output:0Vsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Qsequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
Csequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2TensorListReserveZsequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2/element_shape:output:0Nsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Dsequential_24/bidirectional_24/backward_simple_rnn_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
?sequential_24/bidirectional_24/backward_simple_rnn_11/ReverseV2	ReverseV2Csequential_24/bidirectional_24/backward_simple_rnn_11/transpose:y:0Msequential_24/bidirectional_24/backward_simple_rnn_11/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4¼
ksequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
]sequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorHsequential_24/bidirectional_24/backward_simple_rnn_11/ReverseV2:output:0tsequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Ksequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Msequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Msequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:÷
Esequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_2StridedSliceCsequential_24/bidirectional_24/backward_simple_rnn_11/transpose:y:0Tsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_2/stack:output:0Vsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_1:output:0Vsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_mask
^sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpgsequential_24_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0Ã
Osequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMulMatMulNsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_2:output:0fsequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
_sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOphsequential_24_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ñ
Psequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAddBiasAddYsequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul:product:0gsequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
`sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpisequential_24_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0½
Qsequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1MatMulDsequential_24/bidirectional_24/backward_simple_rnn_11/zeros:output:0hsequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¿
Lsequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/addAddV2Ysequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd:output:0[sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ù
Msequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/TanhTanhPsequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Ssequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Rsequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ç
Esequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1TensorListReserve\sequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0[sequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ|
:sequential_24/bidirectional_24/backward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Nsequential_24/bidirectional_24/backward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Hsequential_24/bidirectional_24/backward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
;sequential_24/bidirectional_24/backward_simple_rnn_11/whileWhileQsequential_24/bidirectional_24/backward_simple_rnn_11/while/loop_counter:output:0Wsequential_24/bidirectional_24/backward_simple_rnn_11/while/maximum_iterations:output:0Csequential_24/bidirectional_24/backward_simple_rnn_11/time:output:0Nsequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1:handle:0Dsequential_24/bidirectional_24/backward_simple_rnn_11/zeros:output:0Nsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_1:output:0msequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0gsequential_24_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resourcehsequential_24_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resourceisequential_24_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
Hsequential_24_bidirectional_24_backward_simple_rnn_11_while_body_5420586*T
condLRJ
Hsequential_24_bidirectional_24_backward_simple_rnn_11_while_cond_5420585*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ·
fsequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ø
Xsequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStackDsequential_24/bidirectional_24/backward_simple_rnn_11/while:output:3osequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
Ksequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Msequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Msequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Esequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_3StridedSliceasequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Tsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_3/stack:output:0Vsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_1:output:0Vsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
Fsequential_24/bidirectional_24/backward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¸
Asequential_24/bidirectional_24/backward_simple_rnn_11/transpose_1	Transposeasequential_24/bidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Osequential_24/bidirectional_24/backward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
*sequential_24/bidirectional_24/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Á
%sequential_24/bidirectional_24/concatConcatV2Msequential_24/bidirectional_24/forward_simple_rnn_11/strided_slice_3:output:0Nsequential_24/bidirectional_24/backward_simple_rnn_11/strided_slice_3:output:03sequential_24/bidirectional_24/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,sequential_24/dense_24/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_24_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¿
sequential_24/dense_24/MatMulMatMul.sequential_24/bidirectional_24/concat:output:04sequential_24/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_24/dense_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_24/dense_24/BiasAddBiasAdd'sequential_24/dense_24/MatMul:product:05sequential_24/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_24/dense_24/SoftmaxSoftmax'sequential_24/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_24/dense_24/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
NoOpNoOp`^sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp_^sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpa^sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp<^sequential_24/bidirectional_24/backward_simple_rnn_11/while_^sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp^^sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp`^sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp;^sequential_24/bidirectional_24/forward_simple_rnn_11/while.^sequential_24/dense_24/BiasAdd/ReadVariableOp-^sequential_24/dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2Â
_sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp_sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp2À
^sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp^sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp2Ä
`sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp`sequential_24/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp2z
;sequential_24/bidirectional_24/backward_simple_rnn_11/while;sequential_24/bidirectional_24/backward_simple_rnn_11/while2À
^sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp^sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp2¾
]sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp]sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp2Â
_sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp_sequential_24/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp2x
:sequential_24/bidirectional_24/forward_simple_rnn_11/while:sequential_24/bidirectional_24/forward_simple_rnn_11/while2^
-sequential_24/dense_24/BiasAdd/ReadVariableOp-sequential_24/dense_24/BiasAdd/ReadVariableOp2\
,sequential_24/dense_24/MatMul/ReadVariableOp,sequential_24/dense_24/MatMul/ReadVariableOp:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_24_input
ó-
Ò
while_body_5424072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_34_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_34/MatMul/ReadVariableOp¢0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_34/BiasAddBiasAdd)while/simple_rnn_cell_34/MatMul:product:07while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_34/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_34/addAddV2)while/simple_rnn_cell_34/BiasAdd:output:0+while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_34/TanhTanh while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_34/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_34/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_34/MatMul/ReadVariableOp1^while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_34_matmul_readvariableop_resource9while_simple_rnn_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_34/MatMul/ReadVariableOp.while/simple_rnn_cell_34/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
while_cond_5424071
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5424071___redundant_placeholder05
1while_while_cond_5424071___redundant_placeholder15
1while_while_cond_5424071___redundant_placeholder25
1while_while_cond_5424071___redundant_placeholder3
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
Gsequential_24_bidirectional_24_forward_simple_rnn_11_while_body_5420478
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_loop_counter
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_maximum_iterationsJ
Fsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholderL
Hsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholder_1L
Hsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholder_2
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_strided_slice_1_0Â
½sequential_24_bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0
nsequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@}
osequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@
psequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@G
Csequential_24_bidirectional_24_forward_simple_rnn_11_while_identityI
Esequential_24_bidirectional_24_forward_simple_rnn_11_while_identity_1I
Esequential_24_bidirectional_24_forward_simple_rnn_11_while_identity_2I
Esequential_24_bidirectional_24_forward_simple_rnn_11_while_identity_3I
Esequential_24_bidirectional_24_forward_simple_rnn_11_while_identity_4
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_strided_slice_1À
»sequential_24_bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor~
lsequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource:4@{
msequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource:@
nsequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢dsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢csequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp¢esequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp½
lsequential_24/bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   °
^sequential_24/bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem½sequential_24_bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0Fsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholderusequential_24/bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0
csequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpnsequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0ä
Tsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMulMatMulesequential_24/bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0ksequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOposequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0à
Usequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAddBiasAdd^sequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul:product:0lsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
esequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOppsequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Ë
Vsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1MatMulHsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholder_2msequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
Qsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/addAddV2^sequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd:output:0`sequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ã
Rsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/TanhTanhUsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
esequential_24/bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Æ
_sequential_24/bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemHsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholder_1nsequential_24/bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:0Vsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ
@sequential_24/bidirectional_24/forward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :û
>sequential_24/bidirectional_24/forward_simple_rnn_11/while/addAddV2Fsequential_24_bidirectional_24_forward_simple_rnn_11_while_placeholderIsequential_24/bidirectional_24/forward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: 
Bsequential_24/bidirectional_24/forward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¼
@sequential_24/bidirectional_24/forward_simple_rnn_11/while/add_1AddV2sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_loop_counterKsequential_24/bidirectional_24/forward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: ø
Csequential_24/bidirectional_24/forward_simple_rnn_11/while/IdentityIdentityDsequential_24/bidirectional_24/forward_simple_rnn_11/while/add_1:z:0@^sequential_24/bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ¿
Esequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity_1Identitysequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_while_maximum_iterations@^sequential_24/bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ø
Esequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity_2IdentityBsequential_24/bidirectional_24/forward_simple_rnn_11/while/add:z:0@^sequential_24/bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ¥
Esequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity_3Identityosequential_24/bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0@^sequential_24/bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
Esequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity_4IdentityVsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0@^sequential_24/bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
?sequential_24/bidirectional_24/forward_simple_rnn_11/while/NoOpNoOpe^sequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpd^sequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpf^sequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Csequential_24_bidirectional_24_forward_simple_rnn_11_while_identityLsequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity:output:0"
Esequential_24_bidirectional_24_forward_simple_rnn_11_while_identity_1Nsequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity_1:output:0"
Esequential_24_bidirectional_24_forward_simple_rnn_11_while_identity_2Nsequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity_2:output:0"
Esequential_24_bidirectional_24_forward_simple_rnn_11_while_identity_3Nsequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity_3:output:0"
Esequential_24_bidirectional_24_forward_simple_rnn_11_while_identity_4Nsequential_24/bidirectional_24/forward_simple_rnn_11/while/Identity_4:output:0"
sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_strided_slice_1sequential_24_bidirectional_24_forward_simple_rnn_11_while_sequential_24_bidirectional_24_forward_simple_rnn_11_strided_slice_1_0"à
msequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resourceosequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"â
nsequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resourcepsequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0"Þ
lsequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resourcensequential_24_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0"þ
»sequential_24_bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor½sequential_24_bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_24_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2Ì
dsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpdsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2Ê
csequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpcsequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp2Î
esequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpesequential_24/bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
while_body_5424292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_34_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_34/MatMul/ReadVariableOp¢0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_34/BiasAddBiasAdd)while/simple_rnn_cell_34/MatMul:product:07while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_34/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_34/addAddV2)while/simple_rnn_cell_34/BiasAdd:output:0+while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_34/TanhTanh while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_34/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_34/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_34/MatMul/ReadVariableOp1^while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_34_matmul_readvariableop_resource9while_simple_rnn_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_34/MatMul/ReadVariableOp.while/simple_rnn_cell_34/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
while_body_5424182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_34_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_34/MatMul/ReadVariableOp¢0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_34/BiasAddBiasAdd)while/simple_rnn_cell_34/MatMul:product:07while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_34/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_34/addAddV2)while/simple_rnn_cell_34/BiasAdd:output:0+while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_34/TanhTanh while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_34/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_34/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_34/MatMul/ReadVariableOp1^while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_34_matmul_readvariableop_resource9while_simple_rnn_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_34/MatMul/ReadVariableOp.while/simple_rnn_cell_34/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
û>
Í
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424139
inputs_0C
1simple_rnn_cell_34_matmul_readvariableop_resource:4@@
2simple_rnn_cell_34_biasadd_readvariableop_resource:@E
3simple_rnn_cell_34_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_34/BiasAdd/ReadVariableOp¢(simple_rnn_cell_34/MatMul/ReadVariableOp¢*simple_rnn_cell_34/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_34/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_34/BiasAddBiasAdd#simple_rnn_cell_34/MatMul:product:01simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_34/MatMul_1MatMulzeros:output:02simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_34/addAddV2#simple_rnn_cell_34/BiasAdd:output:0%simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_34/TanhTanhsimple_rnn_cell_34/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_34_matmul_readvariableop_resource2simple_rnn_cell_34_biasadd_readvariableop_resource3simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
while_body_5424072*
condR
while_cond_5424071*8
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
NoOpNoOp*^simple_rnn_cell_34/BiasAdd/ReadVariableOp)^simple_rnn_cell_34/MatMul/ReadVariableOp+^simple_rnn_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_34/BiasAdd/ReadVariableOp)simple_rnn_cell_34/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_34/MatMul/ReadVariableOp(simple_rnn_cell_34/MatMul/ReadVariableOp2X
*simple_rnn_cell_34/MatMul_1/ReadVariableOp*simple_rnn_cell_34/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ß
¯
while_cond_5420884
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5420884___redundant_placeholder05
1while_while_cond_5420884___redundant_placeholder15
1while_while_cond_5420884___redundant_placeholder25
1while_while_cond_5420884___redundant_placeholder3
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
E__inference_dense_24_layer_call_and_return_conditional_losses_5423985

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
ö©
Ý
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423305
inputs_0Y
Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp¢Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp¢backward_simple_rnn_11/while¢?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp¢@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp¢forward_simple_rnn_11/whileS
forward_simple_rnn_11/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)forward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_11/strided_sliceStridedSlice$forward_simple_rnn_11/Shape:output:02forward_simple_rnn_11/strided_slice/stack:output:04forward_simple_rnn_11/strided_slice/stack_1:output:04forward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_11/zeros/packedPack,forward_simple_rnn_11/strided_slice:output:0-forward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_11/zerosFill+forward_simple_rnn_11/zeros/packed:output:0*forward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
forward_simple_rnn_11/transpose	Transposeinputs_0-forward_simple_rnn_11/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
forward_simple_rnn_11/Shape_1Shape#forward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_11/strided_slice_1StridedSlice&forward_simple_rnn_11/Shape_1:output:04forward_simple_rnn_11/strided_slice_1/stack:output:06forward_simple_rnn_11/strided_slice_1/stack_1:output:06forward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_11/TensorArrayV2TensorListReserve:forward_simple_rnn_11/TensorArrayV2/element_shape:output:0.forward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¢
=forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_11/transpose:y:0Tforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
%forward_simple_rnn_11/strided_slice_2StridedSlice#forward_simple_rnn_11/transpose:y:04forward_simple_rnn_11/strided_slice_2/stack:output:06forward_simple_rnn_11/strided_slice_2/stack_1:output:06forward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_11/simple_rnn_cell_34/MatMulMatMul.forward_simple_rnn_11/strided_slice_2:output:0Fforward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_11/simple_rnn_cell_34/BiasAddBiasAdd9forward_simple_rnn_11/simple_rnn_cell_34/MatMul:product:0Gforward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1MatMul$forward_simple_rnn_11/zeros:output:0Hforward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_11/simple_rnn_cell_34/addAddV29forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd:output:0;forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_11/simple_rnn_cell_34/TanhTanh0forward_simple_rnn_11/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_11/TensorArrayV2_1TensorListReserve<forward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_11/whileWhile1forward_simple_rnn_11/while/loop_counter:output:07forward_simple_rnn_11/while/maximum_iterations:output:0#forward_simple_rnn_11/time:output:0.forward_simple_rnn_11/TensorArrayV2_1:handle:0$forward_simple_rnn_11/zeros:output:0.forward_simple_rnn_11/strided_slice_1:output:0Mforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resourceHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resourceIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_11_while_body_5423128*4
cond,R*
(forward_simple_rnn_11_while_cond_5423127*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_11/while:output:3Oforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_11/strided_slice_3StridedSliceAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_11/strided_slice_3/stack:output:06forward_simple_rnn_11/strided_slice_3/stack_1:output:06forward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_11/transpose_1	TransposeAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
backward_simple_rnn_11/ShapeShapeinputs_0*
T0*
_output_shapes
:t
*backward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_11/strided_sliceStridedSlice%backward_simple_rnn_11/Shape:output:03backward_simple_rnn_11/strided_slice/stack:output:05backward_simple_rnn_11/strided_slice/stack_1:output:05backward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_11/zeros/packedPack-backward_simple_rnn_11/strided_slice:output:0.backward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_11/zerosFill,backward_simple_rnn_11/zeros/packed:output:0+backward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
 backward_simple_rnn_11/transpose	Transposeinputs_0.backward_simple_rnn_11/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr
backward_simple_rnn_11/Shape_1Shape$backward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_11/strided_slice_1StridedSlice'backward_simple_rnn_11/Shape_1:output:05backward_simple_rnn_11/strided_slice_1/stack:output:07backward_simple_rnn_11/strided_slice_1/stack_1:output:07backward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_11/TensorArrayV2TensorListReserve;backward_simple_rnn_11/TensorArrayV2/element_shape:output:0/backward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ë
 backward_simple_rnn_11/ReverseV2	ReverseV2$backward_simple_rnn_11/transpose:y:0.backward_simple_rnn_11/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Lbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿª
>backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_11/ReverseV2:output:0Ubackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
&backward_simple_rnn_11/strided_slice_2StridedSlice$backward_simple_rnn_11/transpose:y:05backward_simple_rnn_11/strided_slice_2/stack:output:07backward_simple_rnn_11/strided_slice_2/stack_1:output:07backward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÈ
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_11/simple_rnn_cell_35/MatMulMatMul/backward_simple_rnn_11/strided_slice_2:output:0Gbackward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_11/simple_rnn_cell_35/BiasAddBiasAdd:backward_simple_rnn_11/simple_rnn_cell_35/MatMul:product:0Hbackward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1MatMul%backward_simple_rnn_11/zeros:output:0Ibackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_11/simple_rnn_cell_35/addAddV2:backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd:output:0<backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_11/simple_rnn_cell_35/TanhTanh1backward_simple_rnn_11/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_11/TensorArrayV2_1TensorListReserve=backward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_11/whileWhile2backward_simple_rnn_11/while/loop_counter:output:08backward_simple_rnn_11/while/maximum_iterations:output:0$backward_simple_rnn_11/time:output:0/backward_simple_rnn_11/TensorArrayV2_1:handle:0%backward_simple_rnn_11/zeros:output:0/backward_simple_rnn_11/strided_slice_1:output:0Nbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resourceIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resourceJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
)backward_simple_rnn_11_while_body_5423236*5
cond-R+
)backward_simple_rnn_11_while_cond_5423235*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_11/while:output:3Pbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_11/strided_slice_3StridedSliceBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_11/strided_slice_3/stack:output:07backward_simple_rnn_11/strided_slice_3/stack_1:output:07backward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_11/transpose_1	TransposeBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_11/strided_slice_3:output:0/backward_simple_rnn_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@^backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpB^backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp^backward_simple_rnn_11/while@^forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?^forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpA^forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp^forward_simple_rnn_11/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp2
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpAbackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp2<
backward_simple_rnn_11/whilebackward_simple_rnn_11/while2
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp2
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp2:
forward_simple_rnn_11/whileforward_simple_rnn_11/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
²
Ñ
(forward_simple_rnn_11_while_cond_5422170H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2J
Fforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5422170___redundant_placeholder0a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5422170___redundant_placeholder1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5422170___redundant_placeholder2a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5422170___redundant_placeholder3(
$forward_simple_rnn_11_while_identity
º
 forward_simple_rnn_11/while/LessLess'forward_simple_rnn_11_while_placeholderFforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_11/while/IdentityIdentity$forward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0*(
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
¯
Ä
8__inference_backward_simple_rnn_11_layer_call_fn_5424480
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421086o
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
»
Â
8__inference_backward_simple_rnn_11_layer_call_fn_5424502

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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421493o
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

ì
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5425068

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
?
Ë
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424469

inputsC
1simple_rnn_cell_34_matmul_readvariableop_resource:4@@
2simple_rnn_cell_34_biasadd_readvariableop_resource:@E
3simple_rnn_cell_34_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_34/BiasAdd/ReadVariableOp¢(simple_rnn_cell_34/MatMul/ReadVariableOp¢*simple_rnn_cell_34/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_34/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_34/BiasAddBiasAdd#simple_rnn_cell_34/MatMul:product:01simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_34/MatMul_1MatMulzeros:output:02simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_34/addAddV2#simple_rnn_cell_34/BiasAdd:output:0%simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_34/TanhTanhsimple_rnn_cell_34/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_34_matmul_readvariableop_resource2simple_rnn_cell_34_biasadd_readvariableop_resource3simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
while_body_5424402*
condR
while_cond_5424401*8
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
NoOpNoOp*^simple_rnn_cell_34/BiasAdd/ReadVariableOp)^simple_rnn_cell_34/MatMul/ReadVariableOp+^simple_rnn_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_34/BiasAdd/ReadVariableOp)simple_rnn_cell_34/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_34/MatMul/ReadVariableOp(simple_rnn_cell_34/MatMul/ReadVariableOp2X
*simple_rnn_cell_34/MatMul_1/ReadVariableOp*simple_rnn_cell_34/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
¯
while_cond_5421708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5421708___redundant_placeholder05
1while_while_cond_5421708___redundant_placeholder15
1while_while_cond_5421708___redundant_placeholder25
1while_while_cond_5421708___redundant_placeholder3
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
Í
ä
)backward_simple_rnn_11_while_cond_5422278J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2L
Hbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5422278___redundant_placeholder0c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5422278___redundant_placeholder1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5422278___redundant_placeholder2c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5422278___redundant_placeholder3)
%backward_simple_rnn_11_while_identity
¾
!backward_simple_rnn_11/while/LessLess(backward_simple_rnn_11_while_placeholderHbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_11/while/IdentityIdentity%backward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0*(
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424961

inputsC
1simple_rnn_cell_35_matmul_readvariableop_resource:4@@
2simple_rnn_cell_35_biasadd_readvariableop_resource:@E
3simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_35/BiasAdd/ReadVariableOp¢(simple_rnn_cell_35/MatMul/ReadVariableOp¢*simple_rnn_cell_35/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_35/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_35/BiasAddBiasAdd#simple_rnn_cell_35/MatMul:product:01simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_35/MatMul_1MatMulzeros:output:02simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_35/addAddV2#simple_rnn_cell_35/BiasAdd:output:0%simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_35/TanhTanhsimple_rnn_cell_35/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_35_matmul_readvariableop_resource2simple_rnn_cell_35_biasadd_readvariableop_resource3simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
while_body_5424894*
condR
while_cond_5424893*8
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
NoOpNoOp*^simple_rnn_cell_35/BiasAdd/ReadVariableOp)^simple_rnn_cell_35/MatMul/ReadVariableOp+^simple_rnn_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_35/BiasAdd/ReadVariableOp)simple_rnn_cell_35/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_35/MatMul/ReadVariableOp(simple_rnn_cell_35/MatMul/ReadVariableOp2X
*simple_rnn_cell_35/MatMul_1/ReadVariableOp*simple_rnn_cell_35/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÏC

)backward_simple_rnn_11_while_body_5421979J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2I
Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_11_while_identity+
'backward_simple_rnn_11_while_identity_1+
'backward_simple_rnn_11_while_identity_2+
'backward_simple_rnn_11_while_identity_3+
'backward_simple_rnn_11_while_identity_4G
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp¢Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
@backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_11_while_placeholderWbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ö
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_11/while/simple_rnn_cell_35/MatMulMatMulGbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAddBiasAdd@backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul:product:0Nbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1MatMul*backward_simple_rnn_11_while_placeholder_2Obackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_11/while/simple_rnn_cell_35/addAddV2@backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd:output:0Bbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_11/while/simple_rnn_cell_35/TanhTanh7backward_simple_rnn_11/while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_11_while_placeholder_1Pbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_11/while/addAddV2(backward_simple_rnn_11_while_placeholder+backward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_11/while/add_1AddV2Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counter-backward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_11/while/IdentityIdentity&backward_simple_rnn_11/while/add_1:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_11/while/Identity_1IdentityLbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_11/while/Identity_2Identity$backward_simple_rnn_11/while/add:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_11/while/Identity_3IdentityQbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_11/while/Identity_4Identity8backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0"^backward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_11/while/NoOpNoOpG^backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpF^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpH^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0"W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0"[
'backward_simple_rnn_11_while_identity_10backward_simple_rnn_11/while/Identity_1:output:0"[
'backward_simple_rnn_11_while_identity_20backward_simple_rnn_11/while/Identity_2:output:0"[
'backward_simple_rnn_11_while_identity_30backward_simple_rnn_11/while/Identity_3:output:0"[
'backward_simple_rnn_11_while_identity_40backward_simple_rnn_11/while/Identity_4:output:0"¤
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resourceQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resourceRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resourcePbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0"
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpFbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpEbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp2
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpGbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
while_body_5420724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_34_5420746_0:4@0
"while_simple_rnn_cell_34_5420748_0:@4
"while_simple_rnn_cell_34_5420750_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_34_5420746:4@.
 while_simple_rnn_cell_34_5420748:@2
 while_simple_rnn_cell_34_5420750:@@¢0while/simple_rnn_cell_34/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_34/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_34_5420746_0"while_simple_rnn_cell_34_5420748_0"while_simple_rnn_cell_34_5420750_0*
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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5420710r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_34/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_34/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_34_5420746"while_simple_rnn_cell_34_5420746_0"F
 while_simple_rnn_cell_34_5420748"while_simple_rnn_cell_34_5420748_0"F
 while_simple_rnn_cell_34_5420750"while_simple_rnn_cell_34_5420750_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_34/StatefulPartitionedCall0while/simple_rnn_cell_34/StatefulPartitionedCall: 
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
ö©
Ý
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423525
inputs_0Y
Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource:4@V
Hforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource:@[
Iforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@Z
Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource:4@W
Ibackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource:@\
Jbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp¢Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp¢backward_simple_rnn_11/while¢?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp¢@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp¢forward_simple_rnn_11/whileS
forward_simple_rnn_11/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)forward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+forward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+forward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#forward_simple_rnn_11/strided_sliceStridedSlice$forward_simple_rnn_11/Shape:output:02forward_simple_rnn_11/strided_slice/stack:output:04forward_simple_rnn_11/strided_slice/stack_1:output:04forward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$forward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@µ
"forward_simple_rnn_11/zeros/packedPack,forward_simple_rnn_11/strided_slice:output:0-forward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!forward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
forward_simple_rnn_11/zerosFill+forward_simple_rnn_11/zeros/packed:output:0*forward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
$forward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
forward_simple_rnn_11/transpose	Transposeinputs_0-forward_simple_rnn_11/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
forward_simple_rnn_11/Shape_1Shape#forward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:u
+forward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%forward_simple_rnn_11/strided_slice_1StridedSlice&forward_simple_rnn_11/Shape_1:output:04forward_simple_rnn_11/strided_slice_1/stack:output:06forward_simple_rnn_11/strided_slice_1/stack_1:output:06forward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1forward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#forward_simple_rnn_11/TensorArrayV2TensorListReserve:forward_simple_rnn_11/TensorArrayV2/element_shape:output:0.forward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Kforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¢
=forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#forward_simple_rnn_11/transpose:y:0Tforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+forward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-forward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
%forward_simple_rnn_11/strided_slice_2StridedSlice#forward_simple_rnn_11/transpose:y:04forward_simple_rnn_11/strided_slice_2/stack:output:06forward_simple_rnn_11/strided_slice_2/stack_1:output:06forward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpGforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0ã
/forward_simple_rnn_11/simple_rnn_cell_34/MatMulMatMul.forward_simple_rnn_11/strided_slice_2:output:0Fforward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ñ
0forward_simple_rnn_11/simple_rnn_cell_34/BiasAddBiasAdd9forward_simple_rnn_11/simple_rnn_cell_34/MatMul:product:0Gforward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Ý
1forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1MatMul$forward_simple_rnn_11/zeros:output:0Hforward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
,forward_simple_rnn_11/simple_rnn_cell_34/addAddV29forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd:output:0;forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-forward_simple_rnn_11/simple_rnn_cell_34/TanhTanh0forward_simple_rnn_11/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3forward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   t
2forward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%forward_simple_rnn_11/TensorArrayV2_1TensorListReserve<forward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0;forward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
forward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.forward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(forward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
forward_simple_rnn_11/whileWhile1forward_simple_rnn_11/while/loop_counter:output:07forward_simple_rnn_11/while/maximum_iterations:output:0#forward_simple_rnn_11/time:output:0.forward_simple_rnn_11/TensorArrayV2_1:handle:0$forward_simple_rnn_11/zeros:output:0.forward_simple_rnn_11/strided_slice_1:output:0Mforward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gforward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resourceHforward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resourceIforward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
(forward_simple_rnn_11_while_body_5423348*4
cond,R*
(forward_simple_rnn_11_while_cond_5423347*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Fforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
8forward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack$forward_simple_rnn_11/while:output:3Oforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements~
+forward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-forward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-forward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%forward_simple_rnn_11/strided_slice_3StridedSliceAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:04forward_simple_rnn_11/strided_slice_3/stack:output:06forward_simple_rnn_11/strided_slice_3/stack_1:output:06forward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask{
&forward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!forward_simple_rnn_11/transpose_1	TransposeAforward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0/forward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
backward_simple_rnn_11/ShapeShapeinputs_0*
T0*
_output_shapes
:t
*backward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,backward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,backward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$backward_simple_rnn_11/strided_sliceStridedSlice%backward_simple_rnn_11/Shape:output:03backward_simple_rnn_11/strided_slice/stack:output:05backward_simple_rnn_11/strided_slice/stack_1:output:05backward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%backward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¸
#backward_simple_rnn_11/zeros/packedPack-backward_simple_rnn_11/strided_slice:output:0.backward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"backward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
backward_simple_rnn_11/zerosFill,backward_simple_rnn_11/zeros/packed:output:0+backward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
%backward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
 backward_simple_rnn_11/transpose	Transposeinputs_0.backward_simple_rnn_11/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr
backward_simple_rnn_11/Shape_1Shape$backward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:v
,backward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&backward_simple_rnn_11/strided_slice_1StridedSlice'backward_simple_rnn_11/Shape_1:output:05backward_simple_rnn_11/strided_slice_1/stack:output:07backward_simple_rnn_11/strided_slice_1/stack_1:output:07backward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2backward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$backward_simple_rnn_11/TensorArrayV2TensorListReserve;backward_simple_rnn_11/TensorArrayV2/element_shape:output:0/backward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%backward_simple_rnn_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ë
 backward_simple_rnn_11/ReverseV2	ReverseV2$backward_simple_rnn_11/transpose:y:0.backward_simple_rnn_11/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Lbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿª
>backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)backward_simple_rnn_11/ReverseV2:output:0Ubackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,backward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.backward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
&backward_simple_rnn_11/strided_slice_2StridedSlice$backward_simple_rnn_11/transpose:y:05backward_simple_rnn_11/strided_slice_2/stack:output:07backward_simple_rnn_11/strided_slice_2/stack_1:output:07backward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÈ
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpHbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0æ
0backward_simple_rnn_11/simple_rnn_cell_35/MatMulMatMul/backward_simple_rnn_11/strided_slice_2:output:0Gbackward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
1backward_simple_rnn_11/simple_rnn_cell_35/BiasAddBiasAdd:backward_simple_rnn_11/simple_rnn_cell_35/MatMul:product:0Hbackward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0à
2backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1MatMul%backward_simple_rnn_11/zeros:output:0Ibackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
-backward_simple_rnn_11/simple_rnn_cell_35/addAddV2:backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd:output:0<backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
.backward_simple_rnn_11/simple_rnn_cell_35/TanhTanh1backward_simple_rnn_11/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
4backward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   u
3backward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
&backward_simple_rnn_11/TensorArrayV2_1TensorListReserve=backward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0<backward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
backward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/backward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)backward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
backward_simple_rnn_11/whileWhile2backward_simple_rnn_11/while/loop_counter:output:08backward_simple_rnn_11/while/maximum_iterations:output:0$backward_simple_rnn_11/time:output:0/backward_simple_rnn_11/TensorArrayV2_1:handle:0%backward_simple_rnn_11/zeros:output:0/backward_simple_rnn_11/strided_slice_1:output:0Nbackward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hbackward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resourceIbackward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resourceJbackward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
)backward_simple_rnn_11_while_body_5423456*5
cond-R+
)backward_simple_rnn_11_while_cond_5423455*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Gbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
9backward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack%backward_simple_rnn_11/while:output:3Pbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
,backward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.backward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.backward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&backward_simple_rnn_11/strided_slice_3StridedSliceBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:05backward_simple_rnn_11/strided_slice_3/stack:output:07backward_simple_rnn_11/strided_slice_3/stack_1:output:07backward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask|
'backward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"backward_simple_rnn_11/transpose_1	TransposeBbackward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:00backward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
concatConcatV2.forward_simple_rnn_11/strided_slice_3:output:0/backward_simple_rnn_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOpA^backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@^backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpB^backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp^backward_simple_rnn_11/while@^forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?^forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpA^forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp^forward_simple_rnn_11/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp@backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp?backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp2
Abackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpAbackward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp2<
backward_simple_rnn_11/whilebackward_simple_rnn_11/while2
?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp?forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp>forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp2
@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp@forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp2:
forward_simple_rnn_11/whileforward_simple_rnn_11/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

î
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5421807

inputs/
forward_simple_rnn_11_5421790:4@+
forward_simple_rnn_11_5421792:@/
forward_simple_rnn_11_5421794:@@0
backward_simple_rnn_11_5421797:4@,
backward_simple_rnn_11_5421799:@0
backward_simple_rnn_11_5421801:@@
identity¢.backward_simple_rnn_11/StatefulPartitionedCall¢-forward_simple_rnn_11/StatefulPartitionedCallË
-forward_simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_11_5421790forward_simple_rnn_11_5421792forward_simple_rnn_11_5421794*
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5421776Ð
.backward_simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_11_5421797backward_simple_rnn_11_5421799backward_simple_rnn_11_5421801*
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421644M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Õ
concatConcatV26forward_simple_rnn_11/StatefulPartitionedCall:output:07backward_simple_rnn_11/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp/^backward_simple_rnn_11/StatefulPartitionedCall.^forward_simple_rnn_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2`
.backward_simple_rnn_11/StatefulPartitionedCall.backward_simple_rnn_11/StatefulPartitionedCall2^
-forward_simple_rnn_11/StatefulPartitionedCall-forward_simple_rnn_11/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
¯
while_cond_5421184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5421184___redundant_placeholder05
1while_while_cond_5421184___redundant_placeholder15
1while_while_cond_5421184___redundant_placeholder25
1while_while_cond_5421184___redundant_placeholder3
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
(forward_simple_rnn_11_while_body_5422171H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2G
Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0
forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_11_while_identity*
&forward_simple_rnn_11_while_identity_1*
&forward_simple_rnn_11_while_identity_2*
&forward_simple_rnn_11_while_identity_3*
&forward_simple_rnn_11_while_identity_4E
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp¢Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
Mforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_11_while_placeholderVforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_11/while/simple_rnn_cell_34/MatMulMatMulFforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAddBiasAdd?forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul:product:0Mforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1MatMul)forward_simple_rnn_11_while_placeholder_2Nforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_11/while/simple_rnn_cell_34/addAddV2?forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd:output:0Aforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_11/while/simple_rnn_cell_34/TanhTanh6forward_simple_rnn_11/while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_11_while_placeholder_1Oforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_11/while/addAddV2'forward_simple_rnn_11_while_placeholder*forward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_11/while/add_1AddV2Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counter,forward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_11/while/IdentityIdentity%forward_simple_rnn_11/while/add_1:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_11/while/Identity_1IdentityJforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_11/while/Identity_2Identity#forward_simple_rnn_11/while/add:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_11/while/Identity_3IdentityPforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_11/while/Identity_4Identity7forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0!^forward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_11/while/NoOpNoOpF^forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpE^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpG^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0"U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0"Y
&forward_simple_rnn_11_while_identity_1/forward_simple_rnn_11/while/Identity_1:output:0"Y
&forward_simple_rnn_11_while_identity_2/forward_simple_rnn_11/while/Identity_2:output:0"Y
&forward_simple_rnn_11_while_identity_3/forward_simple_rnn_11/while/Identity_3:output:0"Y
&forward_simple_rnn_11_while_identity_4/forward_simple_rnn_11/while/Identity_4:output:0"¢
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resourcePforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resourceQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resourceOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0"
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpEforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpDforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp2
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpFforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
while_cond_5420723
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5420723___redundant_placeholder05
1while_while_cond_5420723___redundant_placeholder15
1while_while_cond_5420723___redundant_placeholder25
1while_while_cond_5420723___redundant_placeholder3
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
(forward_simple_rnn_11_while_body_5423568H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2G
Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0
forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_11_while_identity*
&forward_simple_rnn_11_while_identity_1*
&forward_simple_rnn_11_while_identity_2*
&forward_simple_rnn_11_while_identity_3*
&forward_simple_rnn_11_while_identity_4E
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp¢Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
Mforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_11_while_placeholderVforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_11/while/simple_rnn_cell_34/MatMulMatMulFforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAddBiasAdd?forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul:product:0Mforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1MatMul)forward_simple_rnn_11_while_placeholder_2Nforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_11/while/simple_rnn_cell_34/addAddV2?forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd:output:0Aforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_11/while/simple_rnn_cell_34/TanhTanh6forward_simple_rnn_11/while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_11_while_placeholder_1Oforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_11/while/addAddV2'forward_simple_rnn_11_while_placeholder*forward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_11/while/add_1AddV2Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counter,forward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_11/while/IdentityIdentity%forward_simple_rnn_11/while/add_1:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_11/while/Identity_1IdentityJforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_11/while/Identity_2Identity#forward_simple_rnn_11/while/add:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_11/while/Identity_3IdentityPforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_11/while/Identity_4Identity7forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0!^forward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_11/while/NoOpNoOpF^forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpE^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpG^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0"U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0"Y
&forward_simple_rnn_11_while_identity_1/forward_simple_rnn_11/while/Identity_1:output:0"Y
&forward_simple_rnn_11_while_identity_2/forward_simple_rnn_11/while/Identity_2:output:0"Y
&forward_simple_rnn_11_while_identity_3/forward_simple_rnn_11/while/Identity_3:output:0"Y
&forward_simple_rnn_11_while_identity_4/forward_simple_rnn_11/while/Identity_4:output:0"¢
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resourcePforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resourceQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resourceOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0"
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpEforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpDforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp2
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpFforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
while_cond_5424181
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5424181___redundant_placeholder05
1while_while_cond_5424181___redundant_placeholder15
1while_while_cond_5424181___redundant_placeholder25
1while_while_cond_5424181___redundant_placeholder3
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
´
ü
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422408

inputs*
bidirectional_24_5422389:4@&
bidirectional_24_5422391:@*
bidirectional_24_5422393:@@*
bidirectional_24_5422395:4@&
bidirectional_24_5422397:@*
bidirectional_24_5422399:@@#
dense_24_5422402:	
dense_24_5422404:
identity¢(bidirectional_24/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall
(bidirectional_24/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_24_5422389bidirectional_24_5422391bidirectional_24_5422393bidirectional_24_5422395bidirectional_24_5422397bidirectional_24_5422399*
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5422348¡
 dense_24/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_24/StatefulPartitionedCall:output:0dense_24_5422402dense_24_5422404*
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
E__inference_dense_24_layer_call_and_return_conditional_losses_5422073x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_24/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_24/StatefulPartitionedCall(bidirectional_24/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ØC

)backward_simple_rnn_11_while_body_5423456J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2I
Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_11_while_identity+
'backward_simple_rnn_11_while_identity_1+
'backward_simple_rnn_11_while_identity_2+
'backward_simple_rnn_11_while_identity_3+
'backward_simple_rnn_11_while_identity_4G
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp¢Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ£
@backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_11_while_placeholderWbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ö
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_11/while/simple_rnn_cell_35/MatMulMatMulGbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAddBiasAdd@backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul:product:0Nbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1MatMul*backward_simple_rnn_11_while_placeholder_2Obackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_11/while/simple_rnn_cell_35/addAddV2@backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd:output:0Bbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_11/while/simple_rnn_cell_35/TanhTanh7backward_simple_rnn_11/while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_11_while_placeholder_1Pbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_11/while/addAddV2(backward_simple_rnn_11_while_placeholder+backward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_11/while/add_1AddV2Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counter-backward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_11/while/IdentityIdentity&backward_simple_rnn_11/while/add_1:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_11/while/Identity_1IdentityLbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_11/while/Identity_2Identity$backward_simple_rnn_11/while/add:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_11/while/Identity_3IdentityQbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_11/while/Identity_4Identity8backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0"^backward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_11/while/NoOpNoOpG^backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpF^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpH^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0"W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0"[
'backward_simple_rnn_11_while_identity_10backward_simple_rnn_11/while/Identity_1:output:0"[
'backward_simple_rnn_11_while_identity_20backward_simple_rnn_11/while/Identity_2:output:0"[
'backward_simple_rnn_11_while_identity_30backward_simple_rnn_11/while/Identity_3:output:0"[
'backward_simple_rnn_11_while_identity_40backward_simple_rnn_11/while/Identity_4:output:0"¤
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resourceQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resourceRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resourcePbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0"
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpFbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpEbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp2
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpGbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
while_body_5424670
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_35_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_35/MatMul/ReadVariableOp¢0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0¨
.while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_35/BiasAddBiasAdd)while/simple_rnn_cell_35/MatMul:product:07while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_35/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_35/addAddV2)while/simple_rnn_cell_35/BiasAdd:output:0+while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_35/TanhTanh while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_35/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_35/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_35/MatMul/ReadVariableOp1^while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_35_matmul_readvariableop_resource9while_simple_rnn_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_35/MatMul/ReadVariableOp.while/simple_rnn_cell_35/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
ØC

)backward_simple_rnn_11_while_body_5423236J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2I
Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_11_while_identity+
'backward_simple_rnn_11_while_identity_1+
'backward_simple_rnn_11_while_identity_2+
'backward_simple_rnn_11_while_identity_3+
'backward_simple_rnn_11_while_identity_4G
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp¢Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ£
@backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_11_while_placeholderWbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0Ö
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_11/while/simple_rnn_cell_35/MatMulMatMulGbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAddBiasAdd@backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul:product:0Nbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1MatMul*backward_simple_rnn_11_while_placeholder_2Obackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_11/while/simple_rnn_cell_35/addAddV2@backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd:output:0Bbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_11/while/simple_rnn_cell_35/TanhTanh7backward_simple_rnn_11/while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_11_while_placeholder_1Pbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_11/while/addAddV2(backward_simple_rnn_11_while_placeholder+backward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_11/while/add_1AddV2Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counter-backward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_11/while/IdentityIdentity&backward_simple_rnn_11/while/add_1:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_11/while/Identity_1IdentityLbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_11/while/Identity_2Identity$backward_simple_rnn_11/while/add:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_11/while/Identity_3IdentityQbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_11/while/Identity_4Identity8backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0"^backward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_11/while/NoOpNoOpG^backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpF^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpH^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0"W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0"[
'backward_simple_rnn_11_while_identity_10backward_simple_rnn_11/while/Identity_1:output:0"[
'backward_simple_rnn_11_while_identity_20backward_simple_rnn_11/while/Identity_2:output:0"[
'backward_simple_rnn_11_while_identity_30backward_simple_rnn_11/while/Identity_3:output:0"[
'backward_simple_rnn_11_while_identity_40backward_simple_rnn_11/while/Identity_4:output:0"¤
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resourceQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resourceRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resourcePbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0"
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpFbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpEbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp2
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpGbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
¢Ó
ý
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422790

inputsj
Xbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource:4@g
Ybidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource:@l
Zbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@k
Ybidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource:4@h
Zbidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource:@m
[bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@:
'dense_24_matmul_readvariableop_resource:	6
(dense_24_biasadd_readvariableop_resource:
identity¢Qbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Pbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp¢Rbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp¢-bidirectional_24/backward_simple_rnn_11/while¢Pbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Obidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp¢Qbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp¢,bidirectional_24/forward_simple_rnn_11/while¢dense_24/BiasAdd/ReadVariableOp¢dense_24/MatMul/ReadVariableOpb
,bidirectional_24/forward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:
:bidirectional_24/forward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<bidirectional_24/forward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<bidirectional_24/forward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4bidirectional_24/forward_simple_rnn_11/strided_sliceStridedSlice5bidirectional_24/forward_simple_rnn_11/Shape:output:0Cbidirectional_24/forward_simple_rnn_11/strided_slice/stack:output:0Ebidirectional_24/forward_simple_rnn_11/strided_slice/stack_1:output:0Ebidirectional_24/forward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5bidirectional_24/forward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@è
3bidirectional_24/forward_simple_rnn_11/zeros/packedPack=bidirectional_24/forward_simple_rnn_11/strided_slice:output:0>bidirectional_24/forward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:w
2bidirectional_24/forward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    á
,bidirectional_24/forward_simple_rnn_11/zerosFill<bidirectional_24/forward_simple_rnn_11/zeros/packed:output:0;bidirectional_24/forward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5bidirectional_24/forward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
0bidirectional_24/forward_simple_rnn_11/transpose	Transposeinputs>bidirectional_24/forward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
.bidirectional_24/forward_simple_rnn_11/Shape_1Shape4bidirectional_24/forward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:
<bidirectional_24/forward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_24/forward_simple_rnn_11/strided_slice_1StridedSlice7bidirectional_24/forward_simple_rnn_11/Shape_1:output:0Ebidirectional_24/forward_simple_rnn_11/strided_slice_1/stack:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_1:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bbidirectional_24/forward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
4bidirectional_24/forward_simple_rnn_11/TensorArrayV2TensorListReserveKbidirectional_24/forward_simple_rnn_11/TensorArrayV2/element_shape:output:0?bidirectional_24/forward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ­
\bidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Õ
Nbidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor4bidirectional_24/forward_simple_rnn_11/transpose:y:0ebidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<bidirectional_24/forward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>bidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
6bidirectional_24/forward_simple_rnn_11/strided_slice_2StridedSlice4bidirectional_24/forward_simple_rnn_11/transpose:y:0Ebidirectional_24/forward_simple_rnn_11/strided_slice_2/stack:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_1:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskè
Obidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpXbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
@bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMulMatMul?bidirectional_24/forward_simple_rnn_11/strided_slice_2:output:0Wbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Pbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpYbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Abidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAddBiasAddJbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul:product:0Xbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ì
Qbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpZbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Bbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1MatMul5bidirectional_24/forward_simple_rnn_11/zeros:output:0Ybidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
=bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/addAddV2Jbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd:output:0Lbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
>bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/TanhTanhAbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dbidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Cbidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :º
6bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1TensorListReserveMbidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0Lbidirectional_24/forward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
+bidirectional_24/forward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 
?bidirectional_24/forward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ{
9bidirectional_24/forward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø	
,bidirectional_24/forward_simple_rnn_11/whileWhileBbidirectional_24/forward_simple_rnn_11/while/loop_counter:output:0Hbidirectional_24/forward_simple_rnn_11/while/maximum_iterations:output:04bidirectional_24/forward_simple_rnn_11/time:output:0?bidirectional_24/forward_simple_rnn_11/TensorArrayV2_1:handle:05bidirectional_24/forward_simple_rnn_11/zeros:output:0?bidirectional_24/forward_simple_rnn_11/strided_slice_1:output:0^bidirectional_24/forward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Xbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_readvariableop_resourceYbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasadd_readvariableop_resourceZbidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_matmul_1_readvariableop_resource*
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
9bidirectional_24_forward_simple_rnn_11_while_body_5422606*E
cond=R;
9bidirectional_24_forward_simple_rnn_11_while_cond_5422605*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ¨
Wbidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
Ibidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack5bidirectional_24/forward_simple_rnn_11/while:output:3`bidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
<bidirectional_24/forward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
>bidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>bidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ê
6bidirectional_24/forward_simple_rnn_11/strided_slice_3StridedSliceRbidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Ebidirectional_24/forward_simple_rnn_11/strided_slice_3/stack:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_1:output:0Gbidirectional_24/forward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
7bidirectional_24/forward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
2bidirectional_24/forward_simple_rnn_11/transpose_1	TransposeRbidirectional_24/forward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional_24/forward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
-bidirectional_24/backward_simple_rnn_11/ShapeShapeinputs*
T0*
_output_shapes
:
;bidirectional_24/backward_simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=bidirectional_24/backward_simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=bidirectional_24/backward_simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_24/backward_simple_rnn_11/strided_sliceStridedSlice6bidirectional_24/backward_simple_rnn_11/Shape:output:0Dbidirectional_24/backward_simple_rnn_11/strided_slice/stack:output:0Fbidirectional_24/backward_simple_rnn_11/strided_slice/stack_1:output:0Fbidirectional_24/backward_simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6bidirectional_24/backward_simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@ë
4bidirectional_24/backward_simple_rnn_11/zeros/packedPack>bidirectional_24/backward_simple_rnn_11/strided_slice:output:0?bidirectional_24/backward_simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:x
3bidirectional_24/backward_simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ä
-bidirectional_24/backward_simple_rnn_11/zerosFill=bidirectional_24/backward_simple_rnn_11/zeros/packed:output:0<bidirectional_24/backward_simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
6bidirectional_24/backward_simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
1bidirectional_24/backward_simple_rnn_11/transpose	Transposeinputs?bidirectional_24/backward_simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
/bidirectional_24/backward_simple_rnn_11/Shape_1Shape5bidirectional_24/backward_simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:
=bidirectional_24/backward_simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?bidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7bidirectional_24/backward_simple_rnn_11/strided_slice_1StridedSlice8bidirectional_24/backward_simple_rnn_11/Shape_1:output:0Fbidirectional_24/backward_simple_rnn_11/strided_slice_1/stack:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_1:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Cbidirectional_24/backward_simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¬
5bidirectional_24/backward_simple_rnn_11/TensorArrayV2TensorListReserveLbidirectional_24/backward_simple_rnn_11/TensorArrayV2/element_shape:output:0@bidirectional_24/backward_simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
6bidirectional_24/backward_simple_rnn_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ì
1bidirectional_24/backward_simple_rnn_11/ReverseV2	ReverseV25bidirectional_24/backward_simple_rnn_11/transpose:y:0?bidirectional_24/backward_simple_rnn_11/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4®
]bidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   Ý
Obidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor:bidirectional_24/backward_simple_rnn_11/ReverseV2:output:0fbidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=bidirectional_24/backward_simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?bidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:±
7bidirectional_24/backward_simple_rnn_11/strided_slice_2StridedSlice5bidirectional_24/backward_simple_rnn_11/transpose:y:0Fbidirectional_24/backward_simple_rnn_11/strided_slice_2/stack:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_1:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
shrink_axis_maskê
Pbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpYbidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0
Abidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMulMatMul@bidirectional_24/backward_simple_rnn_11/strided_slice_2:output:0Xbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@è
Qbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpZbidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
Bbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAddBiasAddKbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul:product:0Ybidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
Rbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp[bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
Cbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1MatMul6bidirectional_24/backward_simple_rnn_11/zeros:output:0Zbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/addAddV2Kbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd:output:0Mbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@½
?bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/TanhTanhBbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Ebidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
Dbidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :½
7bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1TensorListReserveNbidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/element_shape:output:0Mbidirectional_24/backward_simple_rnn_11/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
,bidirectional_24/backward_simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 
@bidirectional_24/backward_simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ|
:bidirectional_24/backward_simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : å	
-bidirectional_24/backward_simple_rnn_11/whileWhileCbidirectional_24/backward_simple_rnn_11/while/loop_counter:output:0Ibidirectional_24/backward_simple_rnn_11/while/maximum_iterations:output:05bidirectional_24/backward_simple_rnn_11/time:output:0@bidirectional_24/backward_simple_rnn_11/TensorArrayV2_1:handle:06bidirectional_24/backward_simple_rnn_11/zeros:output:0@bidirectional_24/backward_simple_rnn_11/strided_slice_1:output:0_bidirectional_24/backward_simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ybidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_readvariableop_resourceZbidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasadd_readvariableop_resource[bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
:bidirectional_24_backward_simple_rnn_11_while_body_5422714*F
cond>R<
:bidirectional_24_backward_simple_rnn_11_while_cond_5422713*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations ©
Xbidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Î
Jbidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack6bidirectional_24/backward_simple_rnn_11/while:output:3abidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elements
=bidirectional_24/backward_simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
?bidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?bidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
7bidirectional_24/backward_simple_rnn_11/strided_slice_3StridedSliceSbidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Fbidirectional_24/backward_simple_rnn_11/strided_slice_3/stack:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_1:output:0Hbidirectional_24/backward_simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
8bidirectional_24/backward_simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
3bidirectional_24/backward_simple_rnn_11/transpose_1	TransposeSbidirectional_24/backward_simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0Abidirectional_24/backward_simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
bidirectional_24/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
bidirectional_24/concatConcatV2?bidirectional_24/forward_simple_rnn_11/strided_slice_3:output:0@bidirectional_24/backward_simple_rnn_11/strided_slice_3:output:0%bidirectional_24/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_24/MatMulMatMul bidirectional_24/concat:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_24/SoftmaxSoftmaxdense_24/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_24/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOpR^bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpQ^bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpS^bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp.^bidirectional_24/backward_simple_rnn_11/whileQ^bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpP^bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpR^bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp-^bidirectional_24/forward_simple_rnn_11/while ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2¦
Qbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOpQbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/BiasAdd/ReadVariableOp2¤
Pbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOpPbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul/ReadVariableOp2¨
Rbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOpRbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/MatMul_1/ReadVariableOp2^
-bidirectional_24/backward_simple_rnn_11/while-bidirectional_24/backward_simple_rnn_11/while2¤
Pbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOpPbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/BiasAdd/ReadVariableOp2¢
Obidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOpObidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul/ReadVariableOp2¦
Qbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOpQbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/MatMul_1/ReadVariableOp2\
,bidirectional_24/forward_simple_rnn_11/while,bidirectional_24/forward_simple_rnn_11/while2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ÏC

)backward_simple_rnn_11_while_body_5422279J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2I
Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_11_while_identity+
'backward_simple_rnn_11_while_identity_1+
'backward_simple_rnn_11_while_identity_2+
'backward_simple_rnn_11_while_identity_3+
'backward_simple_rnn_11_while_identity_4G
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp¢Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
@backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_11_while_placeholderWbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ö
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_11/while/simple_rnn_cell_35/MatMulMatMulGbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAddBiasAdd@backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul:product:0Nbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1MatMul*backward_simple_rnn_11_while_placeholder_2Obackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_11/while/simple_rnn_cell_35/addAddV2@backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd:output:0Bbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_11/while/simple_rnn_cell_35/TanhTanh7backward_simple_rnn_11/while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_11_while_placeholder_1Pbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_11/while/addAddV2(backward_simple_rnn_11_while_placeholder+backward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_11/while/add_1AddV2Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counter-backward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_11/while/IdentityIdentity&backward_simple_rnn_11/while/add_1:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_11/while/Identity_1IdentityLbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_11/while/Identity_2Identity$backward_simple_rnn_11/while/add:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_11/while/Identity_3IdentityQbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_11/while/Identity_4Identity8backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0"^backward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_11/while/NoOpNoOpG^backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpF^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpH^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0"W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0"[
'backward_simple_rnn_11_while_identity_10backward_simple_rnn_11/while/Identity_1:output:0"[
'backward_simple_rnn_11_while_identity_20backward_simple_rnn_11/while/Identity_2:output:0"[
'backward_simple_rnn_11_while_identity_30backward_simple_rnn_11/while/Identity_3:output:0"[
'backward_simple_rnn_11_while_identity_40backward_simple_rnn_11/while/Identity_4:output:0"¤
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resourceQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resourceRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resourcePbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0"
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpFbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpEbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp2
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpGbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
4__inference_simple_rnn_cell_34_layer_call_fn_5424975

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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5420710o
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
Í
ä
)backward_simple_rnn_11_while_cond_5423895J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2L
Hbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423895___redundant_placeholder0c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423895___redundant_placeholder1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423895___redundant_placeholder2c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423895___redundant_placeholder3)
%backward_simple_rnn_11_while_identity
¾
!backward_simple_rnn_11/while/LessLess(backward_simple_rnn_11_while_placeholderHbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_11/while/IdentityIdentity%backward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0*(
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421249

inputs,
simple_rnn_cell_35_5421172:4@(
simple_rnn_cell_35_5421174:@,
simple_rnn_cell_35_5421176:@@
identity¢*simple_rnn_cell_35/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_35_5421172simple_rnn_cell_35_5421174simple_rnn_cell_35_5421176*
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5421130n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_35_5421172simple_rnn_cell_35_5421174simple_rnn_cell_35_5421176*
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
while_body_5421185*
condR
while_cond_5421184*8
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
NoOpNoOp+^simple_rnn_cell_35/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_35/StatefulPartitionedCall*simple_rnn_cell_35/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ä

J__inference_sequential_24_layer_call_and_return_conditional_losses_5422492
bidirectional_24_input*
bidirectional_24_5422473:4@&
bidirectional_24_5422475:@*
bidirectional_24_5422477:@@*
bidirectional_24_5422479:4@&
bidirectional_24_5422481:@*
bidirectional_24_5422483:@@#
dense_24_5422486:	
dense_24_5422488:
identity¢(bidirectional_24/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall
(bidirectional_24/StatefulPartitionedCallStatefulPartitionedCallbidirectional_24_inputbidirectional_24_5422473bidirectional_24_5422475bidirectional_24_5422477bidirectional_24_5422479bidirectional_24_5422481bidirectional_24_5422483*
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5422348¡
 dense_24/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_24/StatefulPartitionedCall:output:0dense_24_5422486dense_24_5422488*
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
E__inference_dense_24_layer_call_and_return_conditional_losses_5422073x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^bidirectional_24/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ4: : : : : : : : 2T
(bidirectional_24/StatefulPartitionedCall(bidirectional_24/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:c _
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
0
_user_specified_namebidirectional_24_input
Ê

*__inference_dense_24_layer_call_fn_5423974

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
E__inference_dense_24_layer_call_and_return_conditional_losses_5422073o
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


Ó
/__inference_sequential_24_layer_call_fn_5422099
bidirectional_24_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422080o
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
_user_specified_namebidirectional_24_input
ÿ
ê
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5421130

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
4__inference_simple_rnn_cell_35_layer_call_fn_5425051

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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5421130o
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
Ø	
É
%__inference_signature_wrapper_5422521
bidirectional_24_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallbidirectional_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
"__inference__wrapped_model_5420662o
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
_user_specified_namebidirectional_24_input
·	

2__inference_bidirectional_24_layer_call_fn_5423051
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5421807p
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

ì
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5425023

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
while_body_5421185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_35_5421207_0:4@0
"while_simple_rnn_cell_35_5421209_0:@4
"while_simple_rnn_cell_35_5421211_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_35_5421207:4@.
 while_simple_rnn_cell_35_5421209:@2
 while_simple_rnn_cell_35_5421211:@@¢0while/simple_rnn_cell_35/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_35/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_35_5421207_0"while_simple_rnn_cell_35_5421209_0"while_simple_rnn_cell_35_5421211_0*
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5421130r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_35/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_35/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_35_5421207"while_simple_rnn_cell_35_5421207_0"F
 while_simple_rnn_cell_35_5421209"while_simple_rnn_cell_35_5421209_0"F
 while_simple_rnn_cell_35_5421211"while_simple_rnn_cell_35_5421211_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_35/StatefulPartitionedCall0while/simple_rnn_cell_35/StatefulPartitionedCall: 
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
#__inference__traced_restore_5425316
file_prefix3
 assignvariableop_dense_24_kernel:	.
 assignvariableop_1_dense_24_bias:e
Sassignvariableop_2_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel:4@o
]assignvariableop_3_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel:@@_
Qassignvariableop_4_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias:@f
Tassignvariableop_5_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel:4@p
^assignvariableop_6_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel:@@`
Rassignvariableop_7_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias:@&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: =
*assignvariableop_17_adam_dense_24_kernel_m:	6
(assignvariableop_18_adam_dense_24_bias_m:m
[assignvariableop_19_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_m:4@w
eassignvariableop_20_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_m:@@g
Yassignvariableop_21_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_m:@n
\assignvariableop_22_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_m:4@x
fassignvariableop_23_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_m:@@h
Zassignvariableop_24_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_m:@=
*assignvariableop_25_adam_dense_24_kernel_v:	6
(assignvariableop_26_adam_dense_24_bias_v:m
[assignvariableop_27_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_v:4@w
eassignvariableop_28_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_v:@@g
Yassignvariableop_29_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_v:@n
\assignvariableop_30_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_v:4@x
fassignvariableop_31_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_v:@@h
Zassignvariableop_32_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_v:@
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
AssignVariableOpAssignVariableOp assignvariableop_dense_24_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_24_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_2AssignVariableOpSassignvariableop_2_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_3AssignVariableOp]assignvariableop_3_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_4AssignVariableOpQassignvariableop_4_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_5AssignVariableOpTassignvariableop_5_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_6AssignVariableOp^assignvariableop_6_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_7AssignVariableOpRassignvariableop_7_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_24_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_24_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_19AssignVariableOp[assignvariableop_19_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_20AssignVariableOpeassignvariableop_20_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_21AssignVariableOpYassignvariableop_21_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_22AssignVariableOp\assignvariableop_22_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:×
AssignVariableOp_23AssignVariableOpfassignvariableop_23_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_24AssignVariableOpZassignvariableop_24_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_24_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_24_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_27AssignVariableOp[assignvariableop_27_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_28AssignVariableOpeassignvariableop_28_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_29AssignVariableOpYassignvariableop_29_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_30AssignVariableOp\assignvariableop_30_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:×
AssignVariableOp_31AssignVariableOpfassignvariableop_31_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_32AssignVariableOpZassignvariableop_32_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_vIdentity_32:output:0"/device:CPU:0*
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
»
Â
8__inference_backward_simple_rnn_11_layer_call_fn_5424513

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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421644o
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
ÔB
è
(forward_simple_rnn_11_while_body_5421871H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2G
Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0
forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_11_while_identity*
&forward_simple_rnn_11_while_identity_1*
&forward_simple_rnn_11_while_identity_2*
&forward_simple_rnn_11_while_identity_3*
&forward_simple_rnn_11_while_identity_4E
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp¢Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
Mforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_11_while_placeholderVforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_11/while/simple_rnn_cell_34/MatMulMatMulFforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAddBiasAdd?forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul:product:0Mforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1MatMul)forward_simple_rnn_11_while_placeholder_2Nforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_11/while/simple_rnn_cell_34/addAddV2?forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd:output:0Aforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_11/while/simple_rnn_cell_34/TanhTanh6forward_simple_rnn_11/while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_11_while_placeholder_1Oforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_11/while/addAddV2'forward_simple_rnn_11_while_placeholder*forward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_11/while/add_1AddV2Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counter,forward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_11/while/IdentityIdentity%forward_simple_rnn_11/while/add_1:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_11/while/Identity_1IdentityJforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_11/while/Identity_2Identity#forward_simple_rnn_11/while/add:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_11/while/Identity_3IdentityPforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_11/while/Identity_4Identity7forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0!^forward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_11/while/NoOpNoOpF^forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpE^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpG^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0"U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0"Y
&forward_simple_rnn_11_while_identity_1/forward_simple_rnn_11/while/Identity_1:output:0"Y
&forward_simple_rnn_11_while_identity_2/forward_simple_rnn_11/while/Identity_2:output:0"Y
&forward_simple_rnn_11_while_identity_3/forward_simple_rnn_11/while/Identity_3:output:0"Y
&forward_simple_rnn_11_while_identity_4/forward_simple_rnn_11/while/Identity_4:output:0"¢
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resourcePforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resourceQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resourceOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0"
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpEforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpDforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp2
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpFforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
while_cond_5421576
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5421576___redundant_placeholder05
1while_while_cond_5421576___redundant_placeholder15
1while_while_cond_5421576___redundant_placeholder25
1while_while_cond_5421576___redundant_placeholder3
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
¹
Á
7__inference_forward_simple_rnn_11_layer_call_fn_5424029

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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5421776o
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
ü-
Ò
while_body_5421709
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_34_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_34/MatMul/ReadVariableOp¢0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_34/BiasAddBiasAdd)while/simple_rnn_cell_34/MatMul:product:07while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_34/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_34/addAddV2)while/simple_rnn_cell_34/BiasAdd:output:0+while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_34/TanhTanh while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_34/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_34/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_34/MatMul/ReadVariableOp1^while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_34_biasadd_readvariableop_resource:while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_34_matmul_1_readvariableop_resource;while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_34_matmul_readvariableop_resource9while_simple_rnn_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_34/MatMul/ReadVariableOp.while/simple_rnn_cell_34/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp0while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424737
inputs_0C
1simple_rnn_cell_35_matmul_readvariableop_resource:4@@
2simple_rnn_cell_35_biasadd_readvariableop_resource:@E
3simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_35/BiasAdd/ReadVariableOp¢(simple_rnn_cell_35/MatMul/ReadVariableOp¢*simple_rnn_cell_35/MatMul_1/ReadVariableOp¢while=
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
(simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_35/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_35/BiasAddBiasAdd#simple_rnn_cell_35/MatMul:product:01simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_35/MatMul_1MatMulzeros:output:02simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_35/addAddV2#simple_rnn_cell_35/BiasAdd:output:0%simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_35/TanhTanhsimple_rnn_cell_35/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_35_matmul_readvariableop_resource2simple_rnn_cell_35_biasadd_readvariableop_resource3simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
while_body_5424670*
condR
while_cond_5424669*8
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
NoOpNoOp*^simple_rnn_cell_35/BiasAdd/ReadVariableOp)^simple_rnn_cell_35/MatMul/ReadVariableOp+^simple_rnn_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2V
)simple_rnn_cell_35/BiasAdd/ReadVariableOp)simple_rnn_cell_35/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_35/MatMul/ReadVariableOp(simple_rnn_cell_35/MatMul/ReadVariableOp2X
*simple_rnn_cell_35/MatMul_1/ReadVariableOp*simple_rnn_cell_35/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
inputs/0
ÏC

)backward_simple_rnn_11_while_body_5423896J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2I
Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@_
Qbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@d
Rbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@)
%backward_simple_rnn_11_while_identity+
'backward_simple_rnn_11_while_identity_1+
'backward_simple_rnn_11_while_identity_2+
'backward_simple_rnn_11_while_identity_3+
'backward_simple_rnn_11_while_identity_4G
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor`
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource:4@]
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource:@b
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp¢Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
Nbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
@backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0(backward_simple_rnn_11_while_placeholderWbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ö
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpPbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
6backward_simple_rnn_11/while/simple_rnn_cell_35/MatMulMatMulGbackward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Mbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ô
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
7backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAddBiasAdd@backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul:product:0Nbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ñ
8backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1MatMul*backward_simple_rnn_11_while_placeholder_2Obackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
3backward_simple_rnn_11/while/simple_rnn_cell_35/addAddV2@backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd:output:0Bbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
4backward_simple_rnn_11/while/simple_rnn_cell_35/TanhTanh7backward_simple_rnn_11/while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Gbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Î
Abackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*backward_simple_rnn_11_while_placeholder_1Pbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:08backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒd
"backward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 backward_simple_rnn_11/while/addAddV2(backward_simple_rnn_11_while_placeholder+backward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: f
$backward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"backward_simple_rnn_11/while/add_1AddV2Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counter-backward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
%backward_simple_rnn_11/while/IdentityIdentity&backward_simple_rnn_11/while/add_1:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Æ
'backward_simple_rnn_11/while/Identity_1IdentityLbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
'backward_simple_rnn_11/while/Identity_2Identity$backward_simple_rnn_11/while/add:z:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ë
'backward_simple_rnn_11/while/Identity_3IdentityQbackward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ã
'backward_simple_rnn_11/while/Identity_4Identity8backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0"^backward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
!backward_simple_rnn_11/while/NoOpNoOpG^backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpF^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpH^backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Cbackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1Ebackward_simple_rnn_11_while_backward_simple_rnn_11_strided_slice_1_0"W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0"[
'backward_simple_rnn_11_while_identity_10backward_simple_rnn_11/while/Identity_1:output:0"[
'backward_simple_rnn_11_while_identity_20backward_simple_rnn_11/while/Identity_2:output:0"[
'backward_simple_rnn_11_while_identity_30backward_simple_rnn_11/while/Identity_3:output:0"[
'backward_simple_rnn_11_while_identity_40backward_simple_rnn_11/while/Identity_4:output:0"¤
Obackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resourceQbackward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"¦
Pbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resourceRbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"¢
Nbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resourcePbackward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0"
backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Fbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpFbackward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2
Ebackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpEbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp2
Gbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpGbackward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
9bidirectional_24_forward_simple_rnn_11_while_body_5422606j
fbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_loop_counterp
lbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_maximum_iterations<
8bidirectional_24_forward_simple_rnn_11_while_placeholder>
:bidirectional_24_forward_simple_rnn_11_while_placeholder_1>
:bidirectional_24_forward_simple_rnn_11_while_placeholder_2i
ebidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_strided_slice_1_0¦
¡bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0r
`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@o
abidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@t
bbidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@9
5bidirectional_24_forward_simple_rnn_11_while_identity;
7bidirectional_24_forward_simple_rnn_11_while_identity_1;
7bidirectional_24_forward_simple_rnn_11_while_identity_2;
7bidirectional_24_forward_simple_rnn_11_while_identity_3;
7bidirectional_24_forward_simple_rnn_11_while_identity_4g
cbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_strided_slice_1¤
bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorp
^bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource:4@m
_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource:@r
`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢Vbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Ubidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp¢Wbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp¯
^bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ê
Pbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¡bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_08bidirectional_24_forward_simple_rnn_11_while_placeholdergbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ö
Ubidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOp`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0º
Fbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMulMatMulWbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0]bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ô
Vbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpabidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¶
Gbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAddBiasAddPbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul:product:0^bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ú
Wbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpbbidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¡
Hbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1MatMul:bidirectional_24_forward_simple_rnn_11_while_placeholder_2_bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
Cbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/addAddV2Pbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd:output:0Rbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
Dbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/TanhTanhGbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Wbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Qbidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem:bidirectional_24_forward_simple_rnn_11_while_placeholder_1`bidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:0Hbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒt
2bidirectional_24/forward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ñ
0bidirectional_24/forward_simple_rnn_11/while/addAddV28bidirectional_24_forward_simple_rnn_11_while_placeholder;bidirectional_24/forward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: v
4bidirectional_24/forward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
2bidirectional_24/forward_simple_rnn_11/while/add_1AddV2fbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_loop_counter=bidirectional_24/forward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: Î
5bidirectional_24/forward_simple_rnn_11/while/IdentityIdentity6bidirectional_24/forward_simple_rnn_11/while/add_1:z:02^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
7bidirectional_24/forward_simple_rnn_11/while/Identity_1Identitylbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_maximum_iterations2^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Î
7bidirectional_24/forward_simple_rnn_11/while/Identity_2Identity4bidirectional_24/forward_simple_rnn_11/while/add:z:02^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: û
7bidirectional_24/forward_simple_rnn_11/while/Identity_3Identityabidirectional_24/forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ó
7bidirectional_24/forward_simple_rnn_11/while/Identity_4IdentityHbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:02^bidirectional_24/forward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@þ
1bidirectional_24/forward_simple_rnn_11/while/NoOpNoOpW^bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpV^bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpX^bidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ì
cbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_strided_slice_1ebidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_strided_slice_1_0"w
5bidirectional_24_forward_simple_rnn_11_while_identity>bidirectional_24/forward_simple_rnn_11/while/Identity:output:0"{
7bidirectional_24_forward_simple_rnn_11_while_identity_1@bidirectional_24/forward_simple_rnn_11/while/Identity_1:output:0"{
7bidirectional_24_forward_simple_rnn_11_while_identity_2@bidirectional_24/forward_simple_rnn_11/while/Identity_2:output:0"{
7bidirectional_24_forward_simple_rnn_11_while_identity_3@bidirectional_24/forward_simple_rnn_11/while/Identity_3:output:0"{
7bidirectional_24_forward_simple_rnn_11_while_identity_4@bidirectional_24/forward_simple_rnn_11/while/Identity_4:output:0"Ä
_bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resourceabidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"Æ
`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resourcebbidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0"Â
^bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource`bidirectional_24_forward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0"Æ
bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor¡bidirectional_24_forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2°
Vbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpVbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2®
Ubidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpUbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp2²
Wbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpWbidirectional_24/forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
ÉS

:bidirectional_24_backward_simple_rnn_11_while_body_5422714l
hbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_loop_counterr
nbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_maximum_iterations=
9bidirectional_24_backward_simple_rnn_11_while_placeholder?
;bidirectional_24_backward_simple_rnn_11_while_placeholder_1?
;bidirectional_24_backward_simple_rnn_11_while_placeholder_2k
gbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_strided_slice_1_0¨
£bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0s
abidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@p
bbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@u
cbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@:
6bidirectional_24_backward_simple_rnn_11_while_identity<
8bidirectional_24_backward_simple_rnn_11_while_identity_1<
8bidirectional_24_backward_simple_rnn_11_while_identity_2<
8bidirectional_24_backward_simple_rnn_11_while_identity_3<
8bidirectional_24_backward_simple_rnn_11_while_identity_4i
ebidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_strided_slice_1¦
¡bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorq
_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource:4@n
`bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource:@s
abidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢Wbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢Vbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp¢Xbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp°
_bidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ï
Qbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem£bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_09bidirectional_24_backward_simple_rnn_11_while_placeholderhbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0ø
Vbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOpabidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0½
Gbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMulMatMulXbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0^bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ö
Wbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOpbbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0¹
Hbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAddBiasAddQbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul:product:0_bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ü
Xbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOpcbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¤
Ibidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1MatMul;bidirectional_24_backward_simple_rnn_11_while_placeholder_2`bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
Dbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/addAddV2Qbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd:output:0Sbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@É
Ebidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/TanhTanhHbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Xbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Rbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem;bidirectional_24_backward_simple_rnn_11_while_placeholder_1abidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:0Ibidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒu
3bidirectional_24/backward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ô
1bidirectional_24/backward_simple_rnn_11/while/addAddV29bidirectional_24_backward_simple_rnn_11_while_placeholder<bidirectional_24/backward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: w
5bidirectional_24/backward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
3bidirectional_24/backward_simple_rnn_11/while/add_1AddV2hbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_loop_counter>bidirectional_24/backward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: Ñ
6bidirectional_24/backward_simple_rnn_11/while/IdentityIdentity7bidirectional_24/backward_simple_rnn_11/while/add_1:z:03^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
8bidirectional_24/backward_simple_rnn_11/while/Identity_1Identitynbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_maximum_iterations3^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ñ
8bidirectional_24/backward_simple_rnn_11/while/Identity_2Identity5bidirectional_24/backward_simple_rnn_11/while/add:z:03^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: þ
8bidirectional_24/backward_simple_rnn_11/while/Identity_3Identitybbidirectional_24/backward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ö
8bidirectional_24/backward_simple_rnn_11/while/Identity_4IdentityIbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/Tanh:y:03^bidirectional_24/backward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2bidirectional_24/backward_simple_rnn_11/while/NoOpNoOpX^bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpW^bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpY^bidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ð
ebidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_strided_slice_1gbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_strided_slice_1_0"y
6bidirectional_24_backward_simple_rnn_11_while_identity?bidirectional_24/backward_simple_rnn_11/while/Identity:output:0"}
8bidirectional_24_backward_simple_rnn_11_while_identity_1Abidirectional_24/backward_simple_rnn_11/while/Identity_1:output:0"}
8bidirectional_24_backward_simple_rnn_11_while_identity_2Abidirectional_24/backward_simple_rnn_11/while/Identity_2:output:0"}
8bidirectional_24_backward_simple_rnn_11_while_identity_3Abidirectional_24/backward_simple_rnn_11/while/Identity_3:output:0"}
8bidirectional_24_backward_simple_rnn_11_while_identity_4Abidirectional_24/backward_simple_rnn_11/while/Identity_4:output:0"Æ
`bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resourcebbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"È
abidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resourcecbidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"Ä
_bidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resourceabidirectional_24_backward_simple_rnn_11_while_simple_rnn_cell_35_matmul_readvariableop_resource_0"Ê
¡bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor£bidirectional_24_backward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_bidirectional_24_backward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2²
Wbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpWbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2°
Vbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOpVbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul/ReadVariableOp2´
Xbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOpXbidirectional_24/backward_simple_rnn_11/while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
/__inference_sequential_24_layer_call_fn_5422448
bidirectional_24_input
unknown:4@
	unknown_0:@
	unknown_1:@@
	unknown_2:4@
	unknown_3:@
	unknown_4:@@
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422408o
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
_user_specified_namebidirectional_24_input

	
9bidirectional_24_forward_simple_rnn_11_while_cond_5422605j
fbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_loop_counterp
lbidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_maximum_iterations<
8bidirectional_24_forward_simple_rnn_11_while_placeholder>
:bidirectional_24_forward_simple_rnn_11_while_placeholder_1>
:bidirectional_24_forward_simple_rnn_11_while_placeholder_2l
hbidirectional_24_forward_simple_rnn_11_while_less_bidirectional_24_forward_simple_rnn_11_strided_slice_1
bidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_cond_5422605___redundant_placeholder0
bidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_cond_5422605___redundant_placeholder1
bidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_cond_5422605___redundant_placeholder2
bidirectional_24_forward_simple_rnn_11_while_bidirectional_24_forward_simple_rnn_11_while_cond_5422605___redundant_placeholder39
5bidirectional_24_forward_simple_rnn_11_while_identity
þ
1bidirectional_24/forward_simple_rnn_11/while/LessLess8bidirectional_24_forward_simple_rnn_11_while_placeholderhbidirectional_24_forward_simple_rnn_11_while_less_bidirectional_24_forward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: 
5bidirectional_24/forward_simple_rnn_11/while/IdentityIdentity5bidirectional_24/forward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "w
5bidirectional_24_forward_simple_rnn_11_while_identity>bidirectional_24/forward_simple_rnn_11/while/Identity:output:0*(
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
(forward_simple_rnn_11_while_cond_5423347H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2J
Fforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423347___redundant_placeholder0a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423347___redundant_placeholder1a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423347___redundant_placeholder2a
]forward_simple_rnn_11_while_forward_simple_rnn_11_while_cond_5423347___redundant_placeholder3(
$forward_simple_rnn_11_while_identity
º
 forward_simple_rnn_11/while/LessLess'forward_simple_rnn_11_while_placeholderFforward_simple_rnn_11_while_less_forward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: w
$forward_simple_rnn_11/while/IdentityIdentity$forward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0*(
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
)backward_simple_rnn_11_while_cond_5423455J
Fbackward_simple_rnn_11_while_backward_simple_rnn_11_while_loop_counterP
Lbackward_simple_rnn_11_while_backward_simple_rnn_11_while_maximum_iterations,
(backward_simple_rnn_11_while_placeholder.
*backward_simple_rnn_11_while_placeholder_1.
*backward_simple_rnn_11_while_placeholder_2L
Hbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423455___redundant_placeholder0c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423455___redundant_placeholder1c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423455___redundant_placeholder2c
_backward_simple_rnn_11_while_backward_simple_rnn_11_while_cond_5423455___redundant_placeholder3)
%backward_simple_rnn_11_while_identity
¾
!backward_simple_rnn_11/while/LessLess(backward_simple_rnn_11_while_placeholderHbackward_simple_rnn_11_while_less_backward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: y
%backward_simple_rnn_11/while/IdentityIdentity%backward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "W
%backward_simple_rnn_11_while_identity.backward_simple_rnn_11/while/Identity:output:0*(
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
 __inference__traced_save_5425207
file_prefix.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop_
[savev2_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_read_readvariableopi
esavev2_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_read_readvariableop]
Ysavev2_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_read_readvariableop`
\savev2_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_read_readvariableopj
fsavev2_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_read_readvariableop^
Zsavev2_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableopf
bsavev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_m_read_readvariableopp
lsavev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_m_read_readvariableopd
`savev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_m_read_readvariableopg
csavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_m_read_readvariableopq
msavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_m_read_readvariableope
asavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableopf
bsavev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_v_read_readvariableopp
lsavev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_v_read_readvariableopd
`savev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_v_read_readvariableopg
csavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_v_read_readvariableopq
msavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_v_read_readvariableope
asavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop[savev2_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_read_readvariableopesavev2_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_read_readvariableopYsavev2_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_read_readvariableop\savev2_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_read_readvariableopfsavev2_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_read_readvariableopZsavev2_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableopbsavev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_m_read_readvariableoplsavev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_m_read_readvariableop`savev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_m_read_readvariableopcsavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_m_read_readvariableopmsavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_m_read_readvariableopasavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableopbsavev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_kernel_v_read_readvariableoplsavev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_recurrent_kernel_v_read_readvariableop`savev2_adam_bidirectional_24_forward_simple_rnn_11_simple_rnn_cell_34_bias_v_read_readvariableopcsavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_kernel_v_read_readvariableopmsavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_recurrent_kernel_v_read_readvariableopasavev2_adam_bidirectional_24_backward_simple_rnn_11_simple_rnn_cell_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
 5
¬
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5420788

inputs,
simple_rnn_cell_34_5420711:4@(
simple_rnn_cell_34_5420713:@,
simple_rnn_cell_34_5420715:@@
identity¢*simple_rnn_cell_34/StatefulPartitionedCall¢while;
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
*simple_rnn_cell_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_34_5420711simple_rnn_cell_34_5420713simple_rnn_cell_34_5420715*
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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5420710n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_34_5420711simple_rnn_cell_34_5420713simple_rnn_cell_34_5420715*
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
while_body_5420724*
condR
while_cond_5420723*8
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
NoOpNoOp+^simple_rnn_cell_34/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4: : : 2X
*simple_rnn_cell_34/StatefulPartitionedCall*simple_rnn_cell_34/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ß
¯
while_cond_5424291
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5424291___redundant_placeholder05
1while_while_cond_5424291___redundant_placeholder15
1while_while_cond_5424291___redundant_placeholder25
1while_while_cond_5424291___redundant_placeholder3
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
A
Ì
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424849

inputsC
1simple_rnn_cell_35_matmul_readvariableop_resource:4@@
2simple_rnn_cell_35_biasadd_readvariableop_resource:@E
3simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_35/BiasAdd/ReadVariableOp¢(simple_rnn_cell_35/MatMul/ReadVariableOp¢*simple_rnn_cell_35/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_35/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_35/BiasAddBiasAdd#simple_rnn_cell_35/MatMul:product:01simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_35/MatMul_1MatMulzeros:output:02simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_35/addAddV2#simple_rnn_cell_35/BiasAdd:output:0%simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_35/TanhTanhsimple_rnn_cell_35/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_35_matmul_readvariableop_resource2simple_rnn_cell_35_biasadd_readvariableop_resource3simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
while_body_5424782*
condR
while_cond_5424781*8
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
NoOpNoOp*^simple_rnn_cell_35/BiasAdd/ReadVariableOp)^simple_rnn_cell_35/MatMul/ReadVariableOp+^simple_rnn_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_35/BiasAdd/ReadVariableOp)simple_rnn_cell_35/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_35/MatMul/ReadVariableOp(simple_rnn_cell_35/MatMul/ReadVariableOp2X
*simple_rnn_cell_35/MatMul_1/ReadVariableOp*simple_rnn_cell_35/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
Ã
7__inference_forward_simple_rnn_11_layer_call_fn_5423996
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5420788o
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
(forward_simple_rnn_11_while_body_5423788H
Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counterN
Jforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations+
'forward_simple_rnn_11_while_placeholder-
)forward_simple_rnn_11_while_placeholder_1-
)forward_simple_rnn_11_while_placeholder_2G
Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0
forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0:4@^
Pforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0:@c
Qforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0:@@(
$forward_simple_rnn_11_while_identity*
&forward_simple_rnn_11_while_identity_1*
&forward_simple_rnn_11_while_identity_2*
&forward_simple_rnn_11_while_identity_3*
&forward_simple_rnn_11_while_identity_4E
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource:4@\
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource:@a
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource:@@¢Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp¢Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp¢Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp
Mforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   
?forward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0'forward_simple_rnn_11_while_placeholderVforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0Ô
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpReadVariableOpOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0
5forward_simple_rnn_11/while/simple_rnn_cell_34/MatMulMatMulFforward_simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Lforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpReadVariableOpPforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
6forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAddBiasAdd?forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul:product:0Mforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ø
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpReadVariableOpQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0î
7forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1MatMul)forward_simple_rnn_11_while_placeholder_2Nforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2forward_simple_rnn_11/while/simple_rnn_cell_34/addAddV2?forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd:output:0Aforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3forward_simple_rnn_11/while/simple_rnn_cell_34/TanhTanh6forward_simple_rnn_11/while/simple_rnn_cell_34/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Fforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ê
@forward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)forward_simple_rnn_11_while_placeholder_1Oforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem/index:output:07forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒc
!forward_simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_simple_rnn_11/while/addAddV2'forward_simple_rnn_11_while_placeholder*forward_simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: e
#forward_simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!forward_simple_rnn_11/while/add_1AddV2Dforward_simple_rnn_11_while_forward_simple_rnn_11_while_loop_counter,forward_simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 
$forward_simple_rnn_11/while/IdentityIdentity%forward_simple_rnn_11/while/add_1:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Â
&forward_simple_rnn_11/while/Identity_1IdentityJforward_simple_rnn_11_while_forward_simple_rnn_11_while_maximum_iterations!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: 
&forward_simple_rnn_11/while/Identity_2Identity#forward_simple_rnn_11/while/add:z:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: È
&forward_simple_rnn_11/while/Identity_3IdentityPforward_simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^forward_simple_rnn_11/while/NoOp*
T0*
_output_shapes
: À
&forward_simple_rnn_11/while/Identity_4Identity7forward_simple_rnn_11/while/simple_rnn_cell_34/Tanh:y:0!^forward_simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 forward_simple_rnn_11/while/NoOpNoOpF^forward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpE^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpG^forward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Aforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1Cforward_simple_rnn_11_while_forward_simple_rnn_11_strided_slice_1_0"U
$forward_simple_rnn_11_while_identity-forward_simple_rnn_11/while/Identity:output:0"Y
&forward_simple_rnn_11_while_identity_1/forward_simple_rnn_11/while/Identity_1:output:0"Y
&forward_simple_rnn_11_while_identity_2/forward_simple_rnn_11/while/Identity_2:output:0"Y
&forward_simple_rnn_11_while_identity_3/forward_simple_rnn_11/while/Identity_3:output:0"Y
&forward_simple_rnn_11_while_identity_4/forward_simple_rnn_11/while/Identity_4:output:0"¢
Nforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resourcePforward_simple_rnn_11_while_simple_rnn_cell_34_biasadd_readvariableop_resource_0"¤
Oforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resourceQforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_1_readvariableop_resource_0" 
Mforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resourceOforward_simple_rnn_11_while_simple_rnn_cell_34_matmul_readvariableop_resource_0"
}forward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorforward_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Eforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOpEforward_simple_rnn_11/while/simple_rnn_cell_34/BiasAdd/ReadVariableOp2
Dforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOpDforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul/ReadVariableOp2
Fforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOpFforward_simple_rnn_11/while/simple_rnn_cell_34/MatMul_1/ReadVariableOp: 
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5421493

inputsC
1simple_rnn_cell_35_matmul_readvariableop_resource:4@@
2simple_rnn_cell_35_biasadd_readvariableop_resource:@E
3simple_rnn_cell_35_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_35/BiasAdd/ReadVariableOp¢(simple_rnn_cell_35/MatMul/ReadVariableOp¢*simple_rnn_cell_35/MatMul_1/ReadVariableOp¢while;
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
(simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_35_matmul_readvariableop_resource*
_output_shapes

:4@*
dtype0¡
simple_rnn_cell_35/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_35/BiasAddBiasAdd#simple_rnn_cell_35/MatMul:product:01simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_35_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_35/MatMul_1MatMulzeros:output:02simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_35/addAddV2#simple_rnn_cell_35/BiasAdd:output:0%simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_35/TanhTanhsimple_rnn_cell_35/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_35_matmul_readvariableop_resource2simple_rnn_cell_35_biasadd_readvariableop_resource3simple_rnn_cell_35_matmul_1_readvariableop_resource*
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
while_body_5421426*
condR
while_cond_5421425*8
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
NoOpNoOp*^simple_rnn_cell_35/BiasAdd/ReadVariableOp)^simple_rnn_cell_35/MatMul/ReadVariableOp+^simple_rnn_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_35/BiasAdd/ReadVariableOp)simple_rnn_cell_35/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_35/MatMul/ReadVariableOp(simple_rnn_cell_35/MatMul/ReadVariableOp2X
*simple_rnn_cell_35/MatMul_1/ReadVariableOp*simple_rnn_cell_35/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü-
Ò
while_body_5421426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_35_matmul_readvariableop_resource_0:4@H
:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_35_matmul_readvariableop_resource:4@F
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_35/MatMul/ReadVariableOp¢0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_35/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_35_matmul_readvariableop_resource_0*
_output_shapes

:4@*
dtype0Å
while/simple_rnn_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_35/BiasAddBiasAdd)while/simple_rnn_cell_35/MatMul:product:07while/simple_rnn_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_35/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_35/addAddV2)while/simple_rnn_cell_35/BiasAdd:output:0+while/simple_rnn_cell_35/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_35/TanhTanh while/simple_rnn_cell_35/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ò
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_35/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_35/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_35/MatMul/ReadVariableOp1^while/simple_rnn_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_35_biasadd_readvariableop_resource:while_simple_rnn_cell_35_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_35_matmul_1_readvariableop_resource;while_simple_rnn_cell_35_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_35_matmul_readvariableop_resource9while_simple_rnn_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp/while/simple_rnn_cell_35/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_35/MatMul/ReadVariableOp.while/simple_rnn_cell_35/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp0while/simple_rnn_cell_35/MatMul_1/ReadVariableOp: 
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
while_body_5420885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_34_5420907_0:4@0
"while_simple_rnn_cell_34_5420909_0:@4
"while_simple_rnn_cell_34_5420911_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_34_5420907:4@.
 while_simple_rnn_cell_34_5420909:@2
 while_simple_rnn_cell_34_5420911:@@¢0while/simple_rnn_cell_34/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ4   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
element_dtype0®
0while/simple_rnn_cell_34/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_34_5420907_0"while_simple_rnn_cell_34_5420909_0"while_simple_rnn_cell_34_5420911_0*
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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5420832r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_34/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_34/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_34_5420907"while_simple_rnn_cell_34_5420907_0"F
 while_simple_rnn_cell_34_5420909"while_simple_rnn_cell_34_5420909_0"F
 while_simple_rnn_cell_34_5420911"while_simple_rnn_cell_34_5420911_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_34/StatefulPartitionedCall0while/simple_rnn_cell_34/StatefulPartitionedCall: 
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
while_cond_5424401
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_5424401___redundant_placeholder05
1while_while_cond_5424401___redundant_placeholder15
1while_while_cond_5424401___redundant_placeholder25
1while_while_cond_5424401___redundant_placeholder3
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
:bidirectional_24_backward_simple_rnn_11_while_cond_5422940l
hbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_loop_counterr
nbidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_maximum_iterations=
9bidirectional_24_backward_simple_rnn_11_while_placeholder?
;bidirectional_24_backward_simple_rnn_11_while_placeholder_1?
;bidirectional_24_backward_simple_rnn_11_while_placeholder_2n
jbidirectional_24_backward_simple_rnn_11_while_less_bidirectional_24_backward_simple_rnn_11_strided_slice_1
bidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_cond_5422940___redundant_placeholder0
bidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_cond_5422940___redundant_placeholder1
bidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_cond_5422940___redundant_placeholder2
bidirectional_24_backward_simple_rnn_11_while_bidirectional_24_backward_simple_rnn_11_while_cond_5422940___redundant_placeholder3:
6bidirectional_24_backward_simple_rnn_11_while_identity

2bidirectional_24/backward_simple_rnn_11/while/LessLess9bidirectional_24_backward_simple_rnn_11_while_placeholderjbidirectional_24_backward_simple_rnn_11_while_less_bidirectional_24_backward_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: 
6bidirectional_24/backward_simple_rnn_11/while/IdentityIdentity6bidirectional_24/backward_simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "y
6bidirectional_24_backward_simple_rnn_11_while_identity?bidirectional_24/backward_simple_rnn_11/while/Identity:output:0*(
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
/__inference_sequential_24_layer_call_fn_5422563

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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422408o
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
ÿ
ê
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5420832

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
	

2__inference_bidirectional_24_layer_call_fn_5423085

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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5422348p
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
bidirectional_24_inputC
(serving_default_bidirectional_24_input:0ÿÿÿÿÿÿÿÿÿ4<
dense_240
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
/__inference_sequential_24_layer_call_fn_5422099
/__inference_sequential_24_layer_call_fn_5422542
/__inference_sequential_24_layer_call_fn_5422563
/__inference_sequential_24_layer_call_fn_5422448¿
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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422790
J__inference_sequential_24_layer_call_and_return_conditional_losses_5423017
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422470
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422492¿
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
"__inference__wrapped_model_5420662bidirectional_24_input"
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
2__inference_bidirectional_24_layer_call_fn_5423034
2__inference_bidirectional_24_layer_call_fn_5423051
2__inference_bidirectional_24_layer_call_fn_5423068
2__inference_bidirectional_24_layer_call_fn_5423085å
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423305
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423525
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423745
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423965å
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
*__inference_dense_24_layer_call_fn_5423974¢
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
E__inference_dense_24_layer_call_and_return_conditional_losses_5423985¢
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
": 	2dense_24/kernel
:2dense_24/bias
R:P4@2@bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel
\:Z@@2Jbidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel
L:J@2>bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias
S:Q4@2Abidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel
]:[@@2Kbidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel
M:K@2?bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias
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
/__inference_sequential_24_layer_call_fn_5422099bidirectional_24_input"¿
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
/__inference_sequential_24_layer_call_fn_5422542inputs"¿
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
/__inference_sequential_24_layer_call_fn_5422563inputs"¿
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
/__inference_sequential_24_layer_call_fn_5422448bidirectional_24_input"¿
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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422790inputs"¿
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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5423017inputs"¿
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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422470bidirectional_24_input"¿
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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422492bidirectional_24_input"¿
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
%__inference_signature_wrapper_5422521bidirectional_24_input"
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
2__inference_bidirectional_24_layer_call_fn_5423034inputs/0"å
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
2__inference_bidirectional_24_layer_call_fn_5423051inputs/0"å
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
2__inference_bidirectional_24_layer_call_fn_5423068inputs"å
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
2__inference_bidirectional_24_layer_call_fn_5423085inputs"å
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423305inputs/0"å
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423525inputs/0"å
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423745inputs"å
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423965inputs"å
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
7__inference_forward_simple_rnn_11_layer_call_fn_5423996
7__inference_forward_simple_rnn_11_layer_call_fn_5424007
7__inference_forward_simple_rnn_11_layer_call_fn_5424018
7__inference_forward_simple_rnn_11_layer_call_fn_5424029Ô
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424139
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424249
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424359
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424469Ô
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
8__inference_backward_simple_rnn_11_layer_call_fn_5424480
8__inference_backward_simple_rnn_11_layer_call_fn_5424491
8__inference_backward_simple_rnn_11_layer_call_fn_5424502
8__inference_backward_simple_rnn_11_layer_call_fn_5424513Ô
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424625
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424737
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424849
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424961Ô
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
*__inference_dense_24_layer_call_fn_5423974inputs"¢
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
E__inference_dense_24_layer_call_and_return_conditional_losses_5423985inputs"¢
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
7__inference_forward_simple_rnn_11_layer_call_fn_5423996inputs/0"Ô
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
7__inference_forward_simple_rnn_11_layer_call_fn_5424007inputs/0"Ô
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
7__inference_forward_simple_rnn_11_layer_call_fn_5424018inputs"Ô
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
7__inference_forward_simple_rnn_11_layer_call_fn_5424029inputs"Ô
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424139inputs/0"Ô
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424249inputs/0"Ô
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424359inputs"Ô
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424469inputs"Ô
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
4__inference_simple_rnn_cell_34_layer_call_fn_5424975
4__inference_simple_rnn_cell_34_layer_call_fn_5424989½
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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5425006
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5425023½
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
8__inference_backward_simple_rnn_11_layer_call_fn_5424480inputs/0"Ô
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
8__inference_backward_simple_rnn_11_layer_call_fn_5424491inputs/0"Ô
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
8__inference_backward_simple_rnn_11_layer_call_fn_5424502inputs"Ô
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
8__inference_backward_simple_rnn_11_layer_call_fn_5424513inputs"Ô
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424625inputs/0"Ô
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424737inputs/0"Ô
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424849inputs"Ô
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424961inputs"Ô
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
4__inference_simple_rnn_cell_35_layer_call_fn_5425037
4__inference_simple_rnn_cell_35_layer_call_fn_5425051½
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5425068
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5425085½
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
4__inference_simple_rnn_cell_34_layer_call_fn_5424975inputsstates/0"½
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
4__inference_simple_rnn_cell_34_layer_call_fn_5424989inputsstates/0"½
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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5425006inputsstates/0"½
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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5425023inputsstates/0"½
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
4__inference_simple_rnn_cell_35_layer_call_fn_5425037inputsstates/0"½
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
4__inference_simple_rnn_cell_35_layer_call_fn_5425051inputsstates/0"½
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5425068inputsstates/0"½
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5425085inputsstates/0"½
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
':%	2Adam/dense_24/kernel/m
 :2Adam/dense_24/bias/m
W:U4@2GAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/m
a:_@@2QAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/m
Q:O@2EAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/m
X:V4@2HAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/m
b:`@@2RAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/m
R:P@2FAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/m
':%	2Adam/dense_24/kernel/v
 :2Adam/dense_24/bias/v
W:U4@2GAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/kernel/v
a:_@@2QAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/recurrent_kernel/v
Q:O@2EAdam/bidirectional_24/forward_simple_rnn_11/simple_rnn_cell_34/bias/v
X:V4@2HAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/kernel/v
b:`@@2RAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/recurrent_kernel/v
R:P@2FAdam/bidirectional_24/backward_simple_rnn_11/simple_rnn_cell_35/bias/v«
"__inference__wrapped_model_5420662! C¢@
9¢6
41
bidirectional_24_inputÿÿÿÿÿÿÿÿÿ4
ª "3ª0
.
dense_24"
dense_24ÿÿÿÿÿÿÿÿÿÔ
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424625}! O¢L
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424737}! O¢L
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424849! Q¢N
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
S__inference_backward_simple_rnn_11_layer_call_and_return_conditional_losses_5424961! Q¢N
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
8__inference_backward_simple_rnn_11_layer_call_fn_5424480p! O¢L
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
8__inference_backward_simple_rnn_11_layer_call_fn_5424491p! O¢L
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
8__inference_backward_simple_rnn_11_layer_call_fn_5424502r! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@®
8__inference_backward_simple_rnn_11_layer_call_fn_5424513r! Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@à
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423305! \¢Y
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423525! \¢Y
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423745u! C¢@
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
M__inference_bidirectional_24_layer_call_and_return_conditional_losses_5423965u! C¢@
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
2__inference_bidirectional_24_layer_call_fn_5423034! \¢Y
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
2__inference_bidirectional_24_layer_call_fn_5423051! \¢Y
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
2__inference_bidirectional_24_layer_call_fn_5423068h! C¢@
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
2__inference_bidirectional_24_layer_call_fn_5423085h! C¢@
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
E__inference_dense_24_layer_call_and_return_conditional_losses_5423985]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_dense_24_layer_call_fn_5423974P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÓ
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424139}O¢L
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424249}O¢L
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424359Q¢N
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
R__inference_forward_simple_rnn_11_layer_call_and_return_conditional_losses_5424469Q¢N
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
7__inference_forward_simple_rnn_11_layer_call_fn_5423996pO¢L
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
7__inference_forward_simple_rnn_11_layer_call_fn_5424007pO¢L
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
7__inference_forward_simple_rnn_11_layer_call_fn_5424018rQ¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@­
7__inference_forward_simple_rnn_11_layer_call_fn_5424029rQ¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@Ì
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422470~! K¢H
A¢>
41
bidirectional_24_inputÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422492~! K¢H
A¢>
41
bidirectional_24_inputÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
J__inference_sequential_24_layer_call_and_return_conditional_losses_5422790n! ;¢8
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
J__inference_sequential_24_layer_call_and_return_conditional_losses_5423017n! ;¢8
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
/__inference_sequential_24_layer_call_fn_5422099q! K¢H
A¢>
41
bidirectional_24_inputÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
/__inference_sequential_24_layer_call_fn_5422448q! K¢H
A¢>
41
bidirectional_24_inputÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_24_layer_call_fn_5422542a! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_24_layer_call_fn_5422563a! ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ
%__inference_signature_wrapper_5422521! ]¢Z
¢ 
SªP
N
bidirectional_24_input41
bidirectional_24_inputÿÿÿÿÿÿÿÿÿ4"3ª0
.
dense_24"
dense_24ÿÿÿÿÿÿÿÿÿ
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5425006·\¢Y
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
O__inference_simple_rnn_cell_34_layer_call_and_return_conditional_losses_5425023·\¢Y
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
4__inference_simple_rnn_cell_34_layer_call_fn_5424975©\¢Y
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
4__inference_simple_rnn_cell_34_layer_call_fn_5424989©\¢Y
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5425068·! \¢Y
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
O__inference_simple_rnn_cell_35_layer_call_and_return_conditional_losses_5425085·! \¢Y
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
4__inference_simple_rnn_cell_35_layer_call_fn_5425037©! \¢Y
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
4__inference_simple_rnn_cell_35_layer_call_fn_5425051©! \¢Y
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