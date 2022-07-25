[toc]

# IMPALA

**IMPALA是一个基于gRPC的大规模分布式强化学习训练的强化学习框架。**

## gRPC

可以直接看强化学习部分

### 什么是RPC？

- RPC (Remote Procedure Call)即**远程过程调用**，是分布式系统常见的一种通信方法。它允许程序调用另一个地址空间（通常是共享网络的另一台机器上）的过程或函数，而不用程序员显式编码这个远程调用的细节。
- 除 RPC 之外，常见的多系统数据交互方案还有分布式消息队列、HTTP 请求调用、数据库和分布式缓存等。
- 其中 RPC 和 HTTP 调用是没有经过中间件的，它们是端到端系统的直接数据交互。

**简单的说**：

- RPC就是从一台机器（客户端）上通过参数传递的方式调用另一台机器（服务器）上的一个函数或方法（可以统称为服务）并得到返回的结果。
- RPC会隐藏底层的通讯细节（不需要直接处理Socket通讯或Http通讯）。
- 客户端发起请求，服务器返回响应（类似于Http的工作方式）RPC在使用形式上像调用本地函数（或方法）一样去调用远程的函数（或方法）。

### 为什么我们要用RPC?

- RPC 的主要目标是让构建分布式应用更容易，在提供强大的远程调用能力时不损失本地调用的语义简洁性。为实现该目标，RPC 框架需提供一种透明调用机制让使用者不必显式的区分本地调用和远程调用。

### 什么是gRPC?

其实就是谷歌（Google）开源的RPC，故称为gRPC，使用了一种特有是数据格式进行高效通信。我们在强化学习部分讲。

![image-20220324175607796](pictures/image-20220324175607796.png)



## 强化学习部分

gRPC通信框架如下，我把它的强化学习通信框架也画出来了（如下）。

![image-20220324180719621](pictures/image-20220324180719621.png)

![image-20220324175549349](pictures/image-20220324175549349.png)

如上图所示，服务端我们对应的是GPU来训练神经网络（梯度下降），而客户端就是我们的Actor（与环境交互的agent）。

我们可以理解为每一个进程对应一个Actor，Actor与环境做交互产生经验轨迹供Learner学习训练（梯度下降）。

>IMPALA智能体的策略网络调用forward(state)方法时（也就是网上常说的推断）使用CPU进行计算，这个过程也可以放在GPU上。放在GPU是SEED-RL上改进的，但是代码中尽量还原IMPALA代码。

## Why IMPALA
强化学习需要大量在线得到的样本。这就涉及到其中一个问题：产生的样本怎么处理？如果是on-policy的RL算法，样本用过一次就扔掉，十分浪费（千辛万苦采到的样本，就用一次？太浪费了！哪个监督学习里的样本不用个百八十次的~）。如果off-policy的RL算法就可以做到以前的样本反复利用（有RL基础的同学可以马上联想到Deep Q-Learning的Replay Buffer）。

涉及到的第二个问题：如何高效的产生样本？当然是并行或分布式啦！Actor和Learner各干各的，谁都不要等谁。Actor们（不只一个）不断地将采到的样本放到Replay Buffer里，并周期性地从Learner那拿到最新的参数。Learner不断地从Replay Buffer里拿样本过来训练自己的参数。要达到这样的目的，也只有off-policy的方法可以这么干了。而如果是on-policy的方法就要遵循：Actor放样本到buffer->Learner取样本训练参数->Actor取最新的参数->Actor执行动作收集样本->Actor放样本到buffer，这个过程按顺序来的，无法并行。

涉及到的第三个问题：off-policy的方法是可以让样本收集和学习变成并行，还可以利用老样本，但是那些比较老的样本就这么直接拿来更新当前的参数，也会产生利用效率不高的问题（可以理解成并不能有效提升当前的Agent水平）。
### 使用IMALA带来的问题

与A3C不同，在A3C中，worker将关于策略参数传递给中央参数服务器，而Impala Actors将经验轨迹(状态、行动和奖励的序列)传递给集中的Learner。由于Impala的Learner已经获得了完整的经验轨迹，它使用GPU来执行小批量轨迹的更新，并行的收集经验和训练。这种解耦结构可以获得很高的吞吐量，但是由于用于生成轨迹的策略在梯度计算时可能会落后于学习者的策略多次更新，所以学习变成了Off-Policy，因此作者引入了V-trace Off-Policy Actor-Critic算法来纠正这种有害的差异。
### 公式
先上一下V-trace target本尊：
$$
v_{s}=V_{x_{s}}+\sum_{t=s}^{s+n-1} \gamma^{t-s}\left(\Pi_{i=s}^{t-1} c_{i}\right) \delta_{t} V
$$

非常复杂，看不懂，有没有？看看里面一些变量的定义是什么：

$$
\delta_{t} V=\rho_{t}\left(r_{t}+\gamma V_{x_{t+1}}-V_{x_{t}}\right)
$$

括号里的表达式是Temporal Difference，然而外面的 [公式] 是什么含义了？

$$
\rho_{t}=\min \left(\bar{\rho}, \frac{\pi\left(a_{t} \mid x_{t}\right)}{\mu\left(a_{t} \mid x_{t}\right)}\right)
$$

还有其它变量的定义：

$$
\begin{aligned}
&c_{i}=\min \left(\bar{c}, \frac{\pi\left(a_{i} \mid x_{i}\right)}{\mu\left(a_{i} \mid x_{i}\right)}\right) \\
&\bar{\rho} \geq \bar{c}
\end{aligned}
$$

看到上面的表达式，是不是联想起什么了？这就是Importance Sampling。从策略$\mu$中采样，更新当前的策略$\pi$。只不过加上了最大值的限制，不能超过$\bar{\rho}$和$\bar{c}$，也就是说进行了截断处理，一般情况下这两个值可以设置成1啦。


$$
\pi_{\bar{\rho}}(a \mid x)=\frac{\min (\bar{\rho} \mu(a \mid x), \pi(a \mid x))}{\sum_{b \in x} \min (\bar{\rho} \mu(b \mid x), \pi(b \mid x))}
$$

$\pi_{\bar{\rho}}$是一个介于$\bar{\rho}$和$\bar{c}$的中间态的策略。为什么这么定义了？如果当$\bar{\rho}=$无穷，那么$\pi_{\bar{\rho}}$就会变成策略$\pi$，如果$\bar{\rho}=$趋于0（是接近于0，不是等于），那么$\pi_{\bar{\rho}}$就会变成策略$\mu$。（所以$\bar{\rho}$越大，那么off-policy学习的bias就越小，相应的variance就越大。）

$c_i$的乘积表示$\delta_tV$在时刻$t$影响前面时刻$s$的值函数更新的强弱程度。 $\pi$和$mu$差距越大，那么off-poliocy越明显，那么这个乘积的variance就越大。这里用了截断方法来控制这种variance。

$\bar{\rho}$影响的是要收敛到什么样的值函数，而$\bar{c}$影响的是收敛到这个值函数的速度。（突然间出来了值函数，有什么样的策略就有什么样的值函数，二者是对应的，就这么理解吧。）

要注意的是， $v_s$在on-policy的情况下（即$\pi=\mu$），并让$c_i=1$，且$\rho_t=1$，就会变成下式：

$$
v_{s}=V\left(x_{s}\right)+\sum_{t=s}^{s+n-1} \gamma^{t-s}\left(r_{t}+\gamma V\left(x_{t+1}\right)-V\left(x_{t}\right)\right)=\sum_{t=s}^{s+n-1} \gamma^{t-s} r_{t}+\gamma^{n} V\left(x_{s+n}\right)
$$

这个式子是Bellman Target (类似于TD Target)。也就是说式(1)在on-policy的特殊情况下就变成了式(2)。于是，同样的算法可以把off和on两种policy通吃了。

V-trace targets 可以用迭代的方式进行计算（让人联想到back view TD($\lambda$)，实际上在某些条件下确实也可以将它转化到TD($\lambda$)，具体看论文）：

$$
v_{s}=V\left(x_{s}\right)+\delta_{s} V+\gamma c_{s}\left(v_{s+1}-V\left(x_{s+1}\right)\right)
$$

### 学习过程：

1. 每个**Actor单独定期**地从Learner同步参数，然后进行数据收集(s, a, r, s')。

2. 所有Actor收集的数据都会**即时存储到数据采样队列**(queue)里。
3. 队列的数据达到**mini-batch**size时，Learner开始**梯度学习**，并更新其参数。
4. Actor与Learner**互不干扰**，Actor定期从Learner同步参数，Learner定量学习更新参数。
5. Learner也可以是**分布式集群**，这种情况下，Actor需要**从集群同步参数**。
6. Actor一般使用**CPU**，Learner使用**GPU**。

## 特点

**1，**与A2C相比，Actor采集数据**无需等待**，并由GPU快速统一学习。**2，**与A3C相比 ，Actor**无需计算梯度**，只需收集数据，数据吞吐量更大。**3，**与GA3C相比，引入**V-trace**策略纠错，同时接受**更大延迟**，方便大规模分布式部署Actors。**4，**框架拓展方便，支持**多任务**学习。

**5，**但是当场景包含很多终止条件的**Episode**，而又对这些终止(**Terimal**)敏感时，不管是在Actor收集数据时，还是在Learner梯度学习时，**分段处理**长短不一的Episode，都会大大降低IMPALA的性能，影响其流畅性；所以场景最好是Episode不会终止或者是对终止不敏感。



## 参考

[百度](https://baijiahao.baidu.com/s?id=1666816367808221112&wfr=spider&for=pc)

[AlphaStar之IMPALA](https://zhuanlan.zhihu.com/p/56043646)