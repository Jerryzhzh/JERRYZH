Š
Ķ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ĪĀ
¢
yanwing_detection/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!yanwing_detection/conv2d/kernel

3yanwing_detection/conv2d/kernel/Read/ReadVariableOpReadVariableOpyanwing_detection/conv2d/kernel*&
_output_shapes
:*
dtype0

yanwing_detection/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameyanwing_detection/conv2d/bias

1yanwing_detection/conv2d/bias/Read/ReadVariableOpReadVariableOpyanwing_detection/conv2d/bias*
_output_shapes
:*
dtype0
¦
!yanwing_detection/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!yanwing_detection/conv2d_1/kernel

5yanwing_detection/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp!yanwing_detection/conv2d_1/kernel*&
_output_shapes
:*
dtype0

yanwing_detection/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!yanwing_detection/conv2d_1/bias

3yanwing_detection/conv2d_1/bias/Read/ReadVariableOpReadVariableOpyanwing_detection/conv2d_1/bias*
_output_shapes
:*
dtype0
¦
!yanwing_detection/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!yanwing_detection/conv2d_2/kernel

5yanwing_detection/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp!yanwing_detection/conv2d_2/kernel*&
_output_shapes
:
*
dtype0

yanwing_detection/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!yanwing_detection/conv2d_2/bias

3yanwing_detection/conv2d_2/bias/Read/ReadVariableOpReadVariableOpyanwing_detection/conv2d_2/bias*
_output_shapes
:
*
dtype0

yanwing_detection/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
N*/
shared_name yanwing_detection/dense/kernel

2yanwing_detection/dense/kernel/Read/ReadVariableOpReadVariableOpyanwing_detection/dense/kernel* 
_output_shapes
:
N*
dtype0

yanwing_detection/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameyanwing_detection/dense/bias

0yanwing_detection/dense/bias/Read/ReadVariableOpReadVariableOpyanwing_detection/dense/bias*
_output_shapes	
:*
dtype0

 yanwing_detection/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" yanwing_detection/dense_1/kernel

4yanwing_detection/dense_1/kernel/Read/ReadVariableOpReadVariableOp yanwing_detection/dense_1/kernel*
_output_shapes
:	*
dtype0

yanwing_detection/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name yanwing_detection/dense_1/bias

2yanwing_detection/dense_1/bias/Read/ReadVariableOpReadVariableOpyanwing_detection/dense_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
õ!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*°!
value¦!B£! B!
Ź
	conv1
pooling1
	conv2
pooling2
	conv3
pooling3
flatten
d1
	d2

	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
 regularization_losses
!trainable_variables
"	keras_api
h

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
R
)	variables
*regularization_losses
+trainable_variables
,	keras_api
R
-	variables
.regularization_losses
/trainable_variables
0	keras_api
h

1kernel
2bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
h

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
F
0
1
2
3
#4
$5
16
27
78
89
 
F
0
1
2
3
#4
$5
16
27
78
89
­
=non_trainable_variables

	variables
>layer_regularization_losses
?layer_metrics
regularization_losses

@layers
trainable_variables
Ametrics
 
\Z
VARIABLE_VALUEyanwing_detection/conv2d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEyanwing_detection/conv2d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
Bnon_trainable_variables
	variables
Clayer_regularization_losses
Dlayer_metrics
regularization_losses

Elayers
trainable_variables
Fmetrics
 
 
 
­
Gnon_trainable_variables
	variables
Hlayer_regularization_losses
Ilayer_metrics
regularization_losses

Jlayers
trainable_variables
Kmetrics
^\
VARIABLE_VALUE!yanwing_detection/conv2d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEyanwing_detection/conv2d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
Lnon_trainable_variables
	variables
Mlayer_regularization_losses
Nlayer_metrics
regularization_losses

Olayers
trainable_variables
Pmetrics
 
 
 
­
Qnon_trainable_variables
	variables
Rlayer_regularization_losses
Slayer_metrics
 regularization_losses

Tlayers
!trainable_variables
Umetrics
^\
VARIABLE_VALUE!yanwing_detection/conv2d_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEyanwing_detection/conv2d_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
­
Vnon_trainable_variables
%	variables
Wlayer_regularization_losses
Xlayer_metrics
&regularization_losses

Ylayers
'trainable_variables
Zmetrics
 
 
 
­
[non_trainable_variables
)	variables
\layer_regularization_losses
]layer_metrics
*regularization_losses

^layers
+trainable_variables
_metrics
 
 
 
­
`non_trainable_variables
-	variables
alayer_regularization_losses
blayer_metrics
.regularization_losses

clayers
/trainable_variables
dmetrics
XV
VARIABLE_VALUEyanwing_detection/dense/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEyanwing_detection/dense/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
­
enon_trainable_variables
3	variables
flayer_regularization_losses
glayer_metrics
4regularization_losses

hlayers
5trainable_variables
imetrics
ZX
VARIABLE_VALUE yanwing_detection/dense_1/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEyanwing_detection/dense_1/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
­
jnon_trainable_variables
9	variables
klayer_regularization_losses
llayer_metrics
:regularization_losses

mlayers
;trainable_variables
nmetrics
 
 
 
?
0
1
2
3
4
5
6
7
	8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_1Placeholder*1
_output_shapes
:’’’’’’’’’šĄ*
dtype0*&
shape:’’’’’’’’’šĄ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1yanwing_detection/conv2d/kernelyanwing_detection/conv2d/bias!yanwing_detection/conv2d_1/kernelyanwing_detection/conv2d_1/bias!yanwing_detection/conv2d_2/kernelyanwing_detection/conv2d_2/biasyanwing_detection/dense/kernelyanwing_detection/dense/bias yanwing_detection/dense_1/kernelyanwing_detection/dense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *.
f)R'
%__inference_signature_wrapper_1095419
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
æ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3yanwing_detection/conv2d/kernel/Read/ReadVariableOp1yanwing_detection/conv2d/bias/Read/ReadVariableOp5yanwing_detection/conv2d_1/kernel/Read/ReadVariableOp3yanwing_detection/conv2d_1/bias/Read/ReadVariableOp5yanwing_detection/conv2d_2/kernel/Read/ReadVariableOp3yanwing_detection/conv2d_2/bias/Read/ReadVariableOp2yanwing_detection/dense/kernel/Read/ReadVariableOp0yanwing_detection/dense/bias/Read/ReadVariableOp4yanwing_detection/dense_1/kernel/Read/ReadVariableOp2yanwing_detection/dense_1/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *)
f$R"
 __inference__traced_save_1095583
ņ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameyanwing_detection/conv2d/kernelyanwing_detection/conv2d/bias!yanwing_detection/conv2d_1/kernelyanwing_detection/conv2d_1/bias!yanwing_detection/conv2d_2/kernelyanwing_detection/conv2d_2/biasyanwing_detection/dense/kernelyanwing_detection/dense/bias yanwing_detection/dense_1/kernelyanwing_detection/dense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *,
f'R%
#__inference__traced_restore_1095623ö
­
E
)__inference_flatten_layer_call_fn_1095490

inputs
identityĢ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’N* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_10953032
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’N2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’%
:W S
/
_output_shapes
:’’’’’’’’’%

 
_user_specified_nameinputs
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_1095485

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’'  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’N2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’N2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’%
:W S
/
_output_shapes
:’’’’’’’’’%

 
_user_specified_nameinputs
	
­
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1095450

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp„
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’r*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’r2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’r2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’r2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’v:::X T
0
_output_shapes
:’’’’’’’’’v
 
_user_specified_nameinputs
å
|
'__inference_dense_layer_call_fn_1095510

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_10953222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’N::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’N
 
_user_specified_nameinputs
×
ó
%__inference_signature_wrapper_1095419
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *+
f&R$
"__inference__wrapped_model_10951732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:’’’’’’’’’šĄ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:’’’’’’’’’šĄ
!
_user_specified_name	input_1


*__inference_conv2d_2_layer_call_fn_1095479

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’6J
*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_10952802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’6J
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’9M::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’9M
 
_user_specified_nameinputs
ē
~
)__inference_dense_1_layer_call_fn_1095530

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallż
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_10953492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ć#
ė
 __inference__traced_save_1095583
file_prefix>
:savev2_yanwing_detection_conv2d_kernel_read_readvariableop<
8savev2_yanwing_detection_conv2d_bias_read_readvariableop@
<savev2_yanwing_detection_conv2d_1_kernel_read_readvariableop>
:savev2_yanwing_detection_conv2d_1_bias_read_readvariableop@
<savev2_yanwing_detection_conv2d_2_kernel_read_readvariableop>
:savev2_yanwing_detection_conv2d_2_bias_read_readvariableop=
9savev2_yanwing_detection_dense_kernel_read_readvariableop;
7savev2_yanwing_detection_dense_bias_read_readvariableop?
;savev2_yanwing_detection_dense_1_kernel_read_readvariableop=
9savev2_yanwing_detection_dense_1_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c59b4f7b497a42d88ddfe8ba012c0620/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename„
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
value­BŖB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_yanwing_detection_conv2d_kernel_read_readvariableop8savev2_yanwing_detection_conv2d_bias_read_readvariableop<savev2_yanwing_detection_conv2d_1_kernel_read_readvariableop:savev2_yanwing_detection_conv2d_1_bias_read_readvariableop<savev2_yanwing_detection_conv2d_2_kernel_read_readvariableop:savev2_yanwing_detection_conv2d_2_bias_read_readvariableop9savev2_yanwing_detection_dense_kernel_read_readvariableop7savev2_yanwing_detection_dense_bias_read_readvariableop;savev2_yanwing_detection_dense_1_kernel_read_readvariableop9savev2_yanwing_detection_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesr
p: :::::
:
:
N::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:&"
 
_output_shapes
:
N:!

_output_shapes	
::%	!

_output_shapes
:	: 


_output_shapes
::

_output_shapes
: 
“
K
/__inference_max_pooling2d_layer_call_fn_1095185

inputs
identityō
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_10951792
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
®
¬
D__inference_dense_1_layer_call_and_return_conditional_losses_1095349

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
«
C__inference_conv2d_layer_call_and_return_conditional_losses_1095430

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:’’’’’’’’’šĄ:::Y U
1
_output_shapes
:’’’’’’’’’šĄ
 
_user_specified_nameinputs
	
«
C__inference_conv2d_layer_call_and_return_conditional_losses_1095224

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:’’’’’’’’’šĄ:::Y U
1
_output_shapes
:’’’’’’’’’šĄ
 
_user_specified_nameinputs
ģ.
Ę
#__inference__traced_restore_1095623
file_prefix4
0assignvariableop_yanwing_detection_conv2d_kernel4
0assignvariableop_1_yanwing_detection_conv2d_bias8
4assignvariableop_2_yanwing_detection_conv2d_1_kernel6
2assignvariableop_3_yanwing_detection_conv2d_1_bias8
4assignvariableop_4_yanwing_detection_conv2d_2_kernel6
2assignvariableop_5_yanwing_detection_conv2d_2_bias5
1assignvariableop_6_yanwing_detection_dense_kernel3
/assignvariableop_7_yanwing_detection_dense_bias7
3assignvariableop_8_yanwing_detection_dense_1_kernel5
1assignvariableop_9_yanwing_detection_dense_1_bias
identity_11¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9«
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
value­BŖB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesā
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityÆ
AssignVariableOpAssignVariableOp0assignvariableop_yanwing_detection_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1µ
AssignVariableOp_1AssignVariableOp0assignvariableop_1_yanwing_detection_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¹
AssignVariableOp_2AssignVariableOp4assignvariableop_2_yanwing_detection_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3·
AssignVariableOp_3AssignVariableOp2assignvariableop_3_yanwing_detection_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¹
AssignVariableOp_4AssignVariableOp4assignvariableop_4_yanwing_detection_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5·
AssignVariableOp_5AssignVariableOp2assignvariableop_5_yanwing_detection_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¶
AssignVariableOp_6AssignVariableOp1assignvariableop_6_yanwing_detection_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7“
AssignVariableOp_7AssignVariableOp/assignvariableop_7_yanwing_detection_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ø
AssignVariableOp_8AssignVariableOp3assignvariableop_8_yanwing_detection_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¶
AssignVariableOp_9AssignVariableOp1assignvariableop_9_yanwing_detection_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpŗ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
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
Ą(
ē
N__inference_yanwing_detection_layer_call_and_return_conditional_losses_1095366
input_1
conv2d_1095235
conv2d_1095237
conv2d_1_1095263
conv2d_1_1095265
conv2d_2_1095291
conv2d_2_1095293
dense_1095333
dense_1095335
dense_1_1095360
dense_1_1095362
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall”
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1095235conv2d_1095237*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’ģ¼*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_10952242 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’v* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_10951792
max_pooling2d/PartitionedCallÉ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1095263conv2d_1_1095265*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’r*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_10952522"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’9M* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_10951912!
max_pooling2d_1/PartitionedCallŹ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1095291conv2d_2_1095293*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’6J
*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_10952802"
 conv2d_2/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’%
* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10952032!
max_pooling2d_2/PartitionedCallž
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’N* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_10953032
flatten/PartitionedCall¬
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1095333dense_1095335*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_10953222
dense/StatefulPartitionedCall»
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1095360dense_1_1095362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_10953492!
dense_1/StatefulPartitionedCall„
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:’’’’’’’’’šĄ::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Z V
1
_output_shapes
:’’’’’’’’’šĄ
!
_user_specified_name	input_1

f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1095179

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1095191

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

}
(__inference_conv2d_layer_call_fn_1095439

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’ģ¼*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_10952242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:’’’’’’’’’ģ¼2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:’’’’’’’’’šĄ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’šĄ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1095252

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp„
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’r*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’r2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’r2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’r2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’v:::X T
0
_output_shapes
:’’’’’’’’’v
 
_user_specified_nameinputs
Ž<
±
"__inference__wrapped_model_1095173
input_1;
7yanwing_detection_conv2d_conv2d_readvariableop_resource<
8yanwing_detection_conv2d_biasadd_readvariableop_resource=
9yanwing_detection_conv2d_1_conv2d_readvariableop_resource>
:yanwing_detection_conv2d_1_biasadd_readvariableop_resource=
9yanwing_detection_conv2d_2_conv2d_readvariableop_resource>
:yanwing_detection_conv2d_2_biasadd_readvariableop_resource:
6yanwing_detection_dense_matmul_readvariableop_resource;
7yanwing_detection_dense_biasadd_readvariableop_resource<
8yanwing_detection_dense_1_matmul_readvariableop_resource=
9yanwing_detection_dense_1_biasadd_readvariableop_resource
identityą
.yanwing_detection/conv2d/Conv2D/ReadVariableOpReadVariableOp7yanwing_detection_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.yanwing_detection/conv2d/Conv2D/ReadVariableOpņ
yanwing_detection/conv2d/Conv2DConv2Dinput_16yanwing_detection/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼*
paddingVALID*
strides
2!
yanwing_detection/conv2d/Conv2D×
/yanwing_detection/conv2d/BiasAdd/ReadVariableOpReadVariableOp8yanwing_detection_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/yanwing_detection/conv2d/BiasAdd/ReadVariableOpī
 yanwing_detection/conv2d/BiasAddBiasAdd(yanwing_detection/conv2d/Conv2D:output:07yanwing_detection/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼2"
 yanwing_detection/conv2d/BiasAdd­
yanwing_detection/conv2d/ReluRelu)yanwing_detection/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ģ¼2
yanwing_detection/conv2d/Reluų
'yanwing_detection/max_pooling2d/MaxPoolMaxPool+yanwing_detection/conv2d/Relu:activations:0*0
_output_shapes
:’’’’’’’’’v*
ksize
*
paddingVALID*
strides
2)
'yanwing_detection/max_pooling2d/MaxPoolę
0yanwing_detection/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9yanwing_detection_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0yanwing_detection/conv2d_1/Conv2D/ReadVariableOp 
!yanwing_detection/conv2d_1/Conv2DConv2D0yanwing_detection/max_pooling2d/MaxPool:output:08yanwing_detection/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’r*
paddingVALID*
strides
2#
!yanwing_detection/conv2d_1/Conv2DŻ
1yanwing_detection/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:yanwing_detection_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1yanwing_detection/conv2d_1/BiasAdd/ReadVariableOpõ
"yanwing_detection/conv2d_1/BiasAddBiasAdd*yanwing_detection/conv2d_1/Conv2D:output:09yanwing_detection/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’r2$
"yanwing_detection/conv2d_1/BiasAdd²
yanwing_detection/conv2d_1/ReluRelu+yanwing_detection/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’r2!
yanwing_detection/conv2d_1/Reluż
)yanwing_detection/max_pooling2d_1/MaxPoolMaxPool-yanwing_detection/conv2d_1/Relu:activations:0*/
_output_shapes
:’’’’’’’’’9M*
ksize
*
paddingVALID*
strides
2+
)yanwing_detection/max_pooling2d_1/MaxPoolę
0yanwing_detection/conv2d_2/Conv2D/ReadVariableOpReadVariableOp9yanwing_detection_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype022
0yanwing_detection/conv2d_2/Conv2D/ReadVariableOp”
!yanwing_detection/conv2d_2/Conv2DConv2D2yanwing_detection/max_pooling2d_1/MaxPool:output:08yanwing_detection/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’6J
*
paddingVALID*
strides
2#
!yanwing_detection/conv2d_2/Conv2DŻ
1yanwing_detection/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:yanwing_detection_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype023
1yanwing_detection/conv2d_2/BiasAdd/ReadVariableOpō
"yanwing_detection/conv2d_2/BiasAddBiasAdd*yanwing_detection/conv2d_2/Conv2D:output:09yanwing_detection/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’6J
2$
"yanwing_detection/conv2d_2/BiasAdd±
yanwing_detection/conv2d_2/ReluRelu+yanwing_detection/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’6J
2!
yanwing_detection/conv2d_2/Reluż
)yanwing_detection/max_pooling2d_2/MaxPoolMaxPool-yanwing_detection/conv2d_2/Relu:activations:0*/
_output_shapes
:’’’’’’’’’%
*
ksize
*
paddingVALID*
strides
2+
)yanwing_detection/max_pooling2d_2/MaxPool
yanwing_detection/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’'  2!
yanwing_detection/flatten/Constā
!yanwing_detection/flatten/ReshapeReshape2yanwing_detection/max_pooling2d_2/MaxPool:output:0(yanwing_detection/flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’N2#
!yanwing_detection/flatten/Reshape×
-yanwing_detection/dense/MatMul/ReadVariableOpReadVariableOp6yanwing_detection_dense_matmul_readvariableop_resource* 
_output_shapes
:
N*
dtype02/
-yanwing_detection/dense/MatMul/ReadVariableOpą
yanwing_detection/dense/MatMulMatMul*yanwing_detection/flatten/Reshape:output:05yanwing_detection/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2 
yanwing_detection/dense/MatMulÕ
.yanwing_detection/dense/BiasAdd/ReadVariableOpReadVariableOp7yanwing_detection_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.yanwing_detection/dense/BiasAdd/ReadVariableOpā
yanwing_detection/dense/BiasAddBiasAdd(yanwing_detection/dense/MatMul:product:06yanwing_detection/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2!
yanwing_detection/dense/BiasAdd”
yanwing_detection/dense/ReluRelu(yanwing_detection/dense/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
yanwing_detection/dense/ReluÜ
/yanwing_detection/dense_1/MatMul/ReadVariableOpReadVariableOp8yanwing_detection_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype021
/yanwing_detection/dense_1/MatMul/ReadVariableOpå
 yanwing_detection/dense_1/MatMulMatMul*yanwing_detection/dense/Relu:activations:07yanwing_detection/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2"
 yanwing_detection/dense_1/MatMulŚ
0yanwing_detection/dense_1/BiasAdd/ReadVariableOpReadVariableOp9yanwing_detection_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0yanwing_detection/dense_1/BiasAdd/ReadVariableOpé
!yanwing_detection/dense_1/BiasAddBiasAdd*yanwing_detection/dense_1/MatMul:product:08yanwing_detection/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2#
!yanwing_detection/dense_1/BiasAddÆ
!yanwing_detection/dense_1/SigmoidSigmoid*yanwing_detection/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2#
!yanwing_detection/dense_1/Sigmoidy
IdentityIdentity%yanwing_detection/dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:’’’’’’’’’šĄ:::::::::::Z V
1
_output_shapes
:’’’’’’’’’šĄ
!
_user_specified_name	input_1
°
Ŗ
B__inference_dense_layer_call_and_return_conditional_losses_1095322

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
N*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’N:::P L
(
_output_shapes
:’’’’’’’’’N
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1095203

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1095470

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’6J
*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’6J
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’6J
2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’6J
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’9M:::W S
/
_output_shapes
:’’’’’’’’’9M
 
_user_specified_nameinputs
	
­
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1095280

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’6J
*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’6J
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’6J
2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’6J
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’9M:::W S
/
_output_shapes
:’’’’’’’’’9M
 
_user_specified_nameinputs
ø
M
1__inference_max_pooling2d_2_layer_call_fn_1095209

inputs
identityö
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10952032
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
	

3__inference_yanwing_detection_layer_call_fn_1095392
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallš
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *W
fRRP
N__inference_yanwing_detection_layer_call_and_return_conditional_losses_10953662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:’’’’’’’’’šĄ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:’’’’’’’’’šĄ
!
_user_specified_name	input_1
ø
M
1__inference_max_pooling2d_1_layer_call_fn_1095197

inputs
identityö
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_10951912
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_1095303

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’'  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’N2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’N2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’%
:W S
/
_output_shapes
:’’’’’’’’’%

 
_user_specified_nameinputs
°
Ŗ
B__inference_dense_layer_call_and_return_conditional_losses_1095501

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
N*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’N:::P L
(
_output_shapes
:’’’’’’’’’N
 
_user_specified_nameinputs
®
¬
D__inference_dense_1_layer_call_and_return_conditional_losses_1095521

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


*__inference_conv2d_1_layer_call_fn_1095459

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’r*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_10952522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’r2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’v::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’v
 
_user_specified_nameinputs"øL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*µ
serving_default”
E
input_1:
serving_default_input_1:0’’’’’’’’’šĄ<
output_10
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:ĆĘ
¹
	conv1
pooling1
	conv2
pooling2
	conv3
pooling3
flatten
d1
	d2

	variables
regularization_losses
trainable_variables
	keras_api

signatures
o__call__
p_default_save_signature
*q&call_and_return_all_conditional_losses"
_tf_keras_modelū{"class_name": "Yanwing_detection", "name": "yanwing_detection", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Yanwing_detection"}}
ģ	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"Ē
_tf_keras_layer­{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 240, 320, 3]}}
ū
	variables
regularization_losses
trainable_variables
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"ģ
_tf_keras_layerŅ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
š	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"Ė
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 118, 158, 6]}}
’
	variables
 regularization_losses
!trainable_variables
"	keras_api
x__call__
*y&call_and_return_all_conditional_losses"š
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ļ	

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
z__call__
*{&call_and_return_all_conditional_losses"Ź
_tf_keras_layer°{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 57, 77, 8]}}
’
)	variables
*regularization_losses
+trainable_variables
,	keras_api
|__call__
*}&call_and_return_all_conditional_losses"š
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ā
-	variables
.regularization_losses
/trainable_variables
0	keras_api
~__call__
*&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ń

1kernel
2bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
__call__
+&call_and_return_all_conditional_losses"Ź
_tf_keras_layer°{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9990}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 9990]}}
ō

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
__call__
+&call_and_return_all_conditional_losses"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 128]}}
f
0
1
2
3
#4
$5
16
27
78
89"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
#4
$5
16
27
78
89"
trackable_list_wrapper
Ź
=non_trainable_variables

	variables
>layer_regularization_losses
?layer_metrics
regularization_losses

@layers
trainable_variables
Ametrics
o__call__
p_default_save_signature
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
9:72yanwing_detection/conv2d/kernel
+:)2yanwing_detection/conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Bnon_trainable_variables
	variables
Clayer_regularization_losses
Dlayer_metrics
regularization_losses

Elayers
trainable_variables
Fmetrics
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Gnon_trainable_variables
	variables
Hlayer_regularization_losses
Ilayer_metrics
regularization_losses

Jlayers
trainable_variables
Kmetrics
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
;:92!yanwing_detection/conv2d_1/kernel
-:+2yanwing_detection/conv2d_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Lnon_trainable_variables
	variables
Mlayer_regularization_losses
Nlayer_metrics
regularization_losses

Olayers
trainable_variables
Pmetrics
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables
	variables
Rlayer_regularization_losses
Slayer_metrics
 regularization_losses

Tlayers
!trainable_variables
Umetrics
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
;:9
2!yanwing_detection/conv2d_2/kernel
-:+
2yanwing_detection/conv2d_2/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
­
Vnon_trainable_variables
%	variables
Wlayer_regularization_losses
Xlayer_metrics
&regularization_losses

Ylayers
'trainable_variables
Zmetrics
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables
)	variables
\layer_regularization_losses
]layer_metrics
*regularization_losses

^layers
+trainable_variables
_metrics
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
`non_trainable_variables
-	variables
alayer_regularization_losses
blayer_metrics
.regularization_losses

clayers
/trainable_variables
dmetrics
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2:0
N2yanwing_detection/dense/kernel
+:)2yanwing_detection/dense/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
°
enon_trainable_variables
3	variables
flayer_regularization_losses
glayer_metrics
4regularization_losses

hlayers
5trainable_variables
imetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
3:1	2 yanwing_detection/dense_1/kernel
,:*2yanwing_detection/dense_1/bias
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
°
jnon_trainable_variables
9	variables
klayer_regularization_losses
llayer_metrics
:regularization_losses

mlayers
;trainable_variables
nmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
3__inference_yanwing_detection_layer_call_fn_1095392Ė
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *0¢-
+(
input_1’’’’’’’’’šĄ
ź2ē
"__inference__wrapped_model_1095173Ą
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *0¢-
+(
input_1’’’’’’’’’šĄ
”2
N__inference_yanwing_detection_layer_call_and_return_conditional_losses_1095366Ė
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *0¢-
+(
input_1’’’’’’’’’šĄ
Ņ2Ļ
(__inference_conv2d_layer_call_fn_1095439¢
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
annotationsŖ *
 
ķ2ź
C__inference_conv2d_layer_call_and_return_conditional_losses_1095430¢
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
annotationsŖ *
 
2
/__inference_max_pooling2d_layer_call_fn_1095185ą
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
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
²2Æ
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1095179ą
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
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ō2Ń
*__inference_conv2d_1_layer_call_fn_1095459¢
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
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1095450¢
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
annotationsŖ *
 
2
1__inference_max_pooling2d_1_layer_call_fn_1095197ą
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
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
“2±
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1095191ą
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
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ō2Ń
*__inference_conv2d_2_layer_call_fn_1095479¢
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
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1095470¢
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
annotationsŖ *
 
2
1__inference_max_pooling2d_2_layer_call_fn_1095209ą
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
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
“2±
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1095203ą
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
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ó2Š
)__inference_flatten_layer_call_fn_1095490¢
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
annotationsŖ *
 
ī2ė
D__inference_flatten_layer_call_and_return_conditional_losses_1095485¢
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
annotationsŖ *
 
Ń2Ī
'__inference_dense_layer_call_fn_1095510¢
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
annotationsŖ *
 
ģ2é
B__inference_dense_layer_call_and_return_conditional_losses_1095501¢
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
annotationsŖ *
 
Ó2Š
)__inference_dense_1_layer_call_fn_1095530¢
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
annotationsŖ *
 
ī2ė
D__inference_dense_1_layer_call_and_return_conditional_losses_1095521¢
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
annotationsŖ *
 
4B2
%__inference_signature_wrapper_1095419input_1£
"__inference__wrapped_model_1095173}
#$1278:¢7
0¢-
+(
input_1’’’’’’’’’šĄ
Ŗ "3Ŗ0
.
output_1"
output_1’’’’’’’’’·
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1095450n8¢5
.¢+
)&
inputs’’’’’’’’’v
Ŗ ".¢+
$!
0’’’’’’’’’r
 
*__inference_conv2d_1_layer_call_fn_1095459a8¢5
.¢+
)&
inputs’’’’’’’’’v
Ŗ "!’’’’’’’’’rµ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1095470l#$7¢4
-¢*
(%
inputs’’’’’’’’’9M
Ŗ "-¢*
# 
0’’’’’’’’’6J

 
*__inference_conv2d_2_layer_call_fn_1095479_#$7¢4
-¢*
(%
inputs’’’’’’’’’9M
Ŗ " ’’’’’’’’’6J
·
C__inference_conv2d_layer_call_and_return_conditional_losses_1095430p9¢6
/¢,
*'
inputs’’’’’’’’’šĄ
Ŗ "/¢,
%"
0’’’’’’’’’ģ¼
 
(__inference_conv2d_layer_call_fn_1095439c9¢6
/¢,
*'
inputs’’’’’’’’’šĄ
Ŗ ""’’’’’’’’’ģ¼„
D__inference_dense_1_layer_call_and_return_conditional_losses_1095521]780¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 }
)__inference_dense_1_layer_call_fn_1095530P780¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¤
B__inference_dense_layer_call_and_return_conditional_losses_1095501^120¢-
&¢#
!
inputs’’’’’’’’’N
Ŗ "&¢#

0’’’’’’’’’
 |
'__inference_dense_layer_call_fn_1095510Q120¢-
&¢#
!
inputs’’’’’’’’’N
Ŗ "’’’’’’’’’©
D__inference_flatten_layer_call_and_return_conditional_losses_1095485a7¢4
-¢*
(%
inputs’’’’’’’’’%

Ŗ "&¢#

0’’’’’’’’’N
 
)__inference_flatten_layer_call_fn_1095490T7¢4
-¢*
(%
inputs’’’’’’’’’%

Ŗ "’’’’’’’’’Nļ
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1095191R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ē
1__inference_max_pooling2d_1_layer_call_fn_1095197R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’ļ
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1095203R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ē
1__inference_max_pooling2d_2_layer_call_fn_1095209R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’ķ
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1095179R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Å
/__inference_max_pooling2d_layer_call_fn_1095185R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’²
%__inference_signature_wrapper_1095419
#$1278E¢B
¢ 
;Ŗ8
6
input_1+(
input_1’’’’’’’’’šĄ"3Ŗ0
.
output_1"
output_1’’’’’’’’’Į
N__inference_yanwing_detection_layer_call_and_return_conditional_losses_1095366o
#$1278:¢7
0¢-
+(
input_1’’’’’’’’’šĄ
Ŗ "%¢"

0’’’’’’’’’
 
3__inference_yanwing_detection_layer_call_fn_1095392b
#$1278:¢7
0¢-
+(
input_1’’’’’’’’’šĄ
Ŗ "’’’’’’’’’