### 15、复制

- 进行复制中的主从服务器双方的数据库将保存相同的数据，概念上将这种现象称作“数据库状态一致”，或者简称“一致”。

- 旧版复制功能的实现

  - Redis的复制功能分为同步( sync）和命令传播( command propagate)两个操作
  - **同步**操作用于将从服务器的数据库状态更新至主服务器当前所处的数据库状态。
    - 当客户端向从服务器发送SLAVEOF命令，要求从服务器复制主服务器时，从服务器首先需要执行同步操作，从服务器对主服务器的同步操作需要通过向主服务器发送SYNC命令来完成，SYNC命令的执行步骤：
      - ![image-20220415090639553](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20220415090639553.png)
  - **命令传播**操作则用于在主服务器的数据库状态被修改，导致主从服务器的数据库状态出现不一致时，让主从服务器的数据库重新回到一致状态。
    - 主服务器会将自己执行的写命令，也即是造成主从服务器不一致的那条写命令，发送给从服务器执行.当从服务器执行了相同的写命令之后，主从服务器将再次回到一致状态。

- 旧版复制功能的缺陷

  - 从服务器对主服务器的复制分为以下两种

  - 初次复制：从服务器以前没有复制过任何主服务器，或者从服务器当前要复制的主服务器和上一次复制的主服务器不同。
  - 断线后重复制：处于命令传播阶段的主从服务器因为网络原因而中断了复制，但从服务器通过自动重连接重新连上了主服务器，并继续复制主服务器。
    - 缺点：效率低（比如主从服务器执行到set k1000 v1000-set k1003 v1003出错，那么从服务器会执行SYNC，这时RDB文件中会包含从k1到k1003所有的指令，做了很多没用的事情）

- 新版复制功能的实现
  - redis 2.8开始使用PSYNC代替SYNC，PSYNC命令具有完整重同步（和SYNC功能一样）和部分重同步（处理断线后重复制，把断线期间的命令发送给从服务器）两种模式
  - ![image-20220415092433775](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20220415092433775.png)

- 部分重同步的实现
  - 主服务器的复制偏移量（ replication offset）和从服务器的复制偏移量。
  - 主服务器的复制积压缓冲区( replication backlog )。
    - 复制积压缓冲区是由主服务器维护的一个固定长度( fixed-size )先进先出（FIFO)队列，默认大小为1MB。
    - 当主服务器进行命令传播时，它不仅会将写命令发送给所有从服务器，还会将写命令入队到复制积压缓冲区中，并且会为每个字节记录复制偏移量
    - 当从服务器重新连上主服务器时，从服务器会通过PSYNC命令将自己的复制偏移量offset发送给主服务器，主服务器会根据这个**复制偏移量**来决定对从服务器执行何种同步操作：
      - 如果 offset偏移量之后的数据（也即是偏移量 offset+1开始的数据）仍然存在于复制积压缓冲区里面，那么主服务器将对从服务器执行部分重同步操作。
      - 相反，如果offset偏移量之后的数据已经不存在于复制积压缓冲区，那么主服务器将对从服务器执行完整重同步操作。
  - 服务器的运行ID ( run ID )。
    - 每个Redis服务器，不论主服务器还是从服务，都会有自己的运行ID。运行ID在服务器启动时自动生成，由40个随机的十六进制字符组成
    - 根据ID来判断执行部分重同步还是完全重同步
- PSYNC命令的实现
  - 如果从服务器没有复制过任何主服务器的内容，会发送：PSYNC ? -1
  - 如果已经复制过某个主服务器，会发送：PSYNC  < RUNID > < OFFSET >
  - 主服务器返回+FULLRESYNC < runid > < offset >：执行完整重同步
  - 主服务器返回+CONTINUE：执行部分重同步
  - 主服务器返回-ERR：表示不支持PSYNC，会执行SYNC
  - ![image-20220415094515064](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20220415094515064.png)

- 复制的实现

  - 通过向从服务器发送SLAVEOF命令，我们可以让一个从服务器去复制一个主服务器

  1. 设置主服务器的地址和端口
  2. 建立套接字连接
  3. 发送PING命令
     1. ![image-20220415095040413](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20220415095040413.png)
  4. 身份验证
  5. 发送端口信息：向主服务器发送从服务器的监听端口号
  6. 同步：在同步操作执行之后，主从服务器双方都是对方的客户端
  7. 命令传播

- 心跳检测

  - 在命令传播阶段，从服务器默认会以每秒一次的频率，向主服务器发送命令：`REPLCONF ACK <replication_offset>`，发送此命令对主从服务器有以下三个作用：
    - 检测主从服务器的网络连接状态。
    - 辅助实现min-slaves选项。
    - 检测命令丢失。

### 16、哨兵

- Sentinel（哨兵）：是redis高可用的解决方案，可以监视主从服务器，当主服务器下线的时候，可以指定一个从服务器当作主服务器
- 启动并初始化Sentinel
  - 启动命令：`redis-sentinel /myredis/sentinel.conf`，执行步骤如下：
  - 初始化服务器。
    - sentinel可以看成是运行在特殊模式下的redis服务器，和普通的redis服务器初始化大致相同。但也有不同：比如不会加载AOF和RDB文件
  - 将普通Redis服务器使用的代码替换成Sentinel专用代码。
    - 启动Sentinel的第二个步骤就是将一部分普通Redis服务器使用的代码替换成Sentinel专用代码。客户端对Sentinel可以执行的命令只有7个
  - 初始化Sentinel状态。
    - Sentinel状态中的masters字典记录了所有被Sentinel监视的主服务器的相关信息
  - 根据给定的配置文件，初始化Sentinel的监视主服务器列表。
  - 创建连向主服务器的网络连接。
    - 初始化Sentinel的最后一步是创建连向被监视主服务器的网络连接，Sentinel将成为主服务器的客户端，它可以向主服务器发送命令，并从命令回复中获取相关的信息。
    - 对于每个被Sentinel监视的主服务器来说，Sentinel 会创建两个连向主服务器的异步网络连接：
      - 一个是**命令连接**，这个连接专门用于向主服务器发送命令，并接收命令回复。
      - 另一个是**订阅连接**，这个连接专门用于订阅主服务器的_sentinel_:hello频道。

- 通过命令连接和订阅连接来与被监视主服务器进行通信？
  - Sentinel默认会以每十秒一次的频率，通过命令连接向被监视的主服务器发送 INFO命令，并通过分析INFO命令的回复来获取主服务器的当前信息（包括主服务器id和从服务器的信息）

- 获取从服务器信息
  - 当Sentinel发现主服务器有新的从服务器出现时，Sentinel除了会为这个新的从服务器创建相应的实例结构之外，Sentinel还会创建连接到从服务器的命令连接和订阅连接。

- 向主服务器和从服务器发送消息
  - 在默认情况下，Sentinel 会以每两秒一次的频率，通过命令连接向所有被监视的主服务器和从服务器发送命令

- 接收来自主从服务器的频道信息
  - 这也就是说，对于每个与Sentinel连接的服务器，Sentinel既通过命令连接向服务器的sentinel_:hello频道发送信息，又通过订阅连接从服务器的 sentinel :hello频道接收信息
  - 更新Sentinel字典：Sentinel为主服务器创建的实例结构中的sentinels字典保存了除Sentinel本身之外，所有同样监视这个主服务器的其他 Sentinel的资料
  - 创建连向其他Sentinel的命令连接：当Sentinel通过频道信息发现一个新的Sentinel时,它不仅会为新Sentinel在sentinels字典中创建相应的实例结构，还会创建一个连向新Sentinel的命令连接，而新Sentinel也同样会创建连向这个Sentinel的命令连接,最终监视同一主服务器的多个Sentinel将形成相互连接的网络
- 检测主观下线状态
  - 在默认情况下，Sentinel会以每秒一次的频率向所有与它创建了命令连接的实例（包括主服务器、从服务器、其他Sentinel在内）发送PING命令，并通过实例返回的PING命令回复判断实例是否在线
- 检查客观下线状态
  - 当Sentinel将一个主服务器判断为主观下线之后，为了确认这个主服务器是否真的下线了，它会向同样监视这一主服务器的其他Sentinel进行询问，看它们是否也认为主服务器已经进入了下线状态（可以是主观下线或者客观下线)。当Sentinel从其他Sentinel那里接收到足够数量的已下线判断之后，Sentinel就会将从服务器判定为客观下线，并对主服务器执行故障转移操作。

- 选举领头Sentinel
  - 当一个主服务器被判断为客观下线时，监视这个下线主服务器的各个 Sentinel会进行协商，选举出一个领头Sentinel，并由领头Sentinel对下线主服务器执行故障转移操作。

- 故障转移
  - 1)在已下线主服务器属下的所有从服务器里面，挑选出一个从服务器，并将其转换为主服务器。
  - 2)让已下线主服务器属下的所有从服务器改为复制新的主服务器。
  - 3）将已下线主服务器设置为新的主服务器的从服务器，当这个旧的主服务器重新上线时，它就会成为新的主服务器的从服务器。

### 17、集群

- Redis集群是Redis提供的分布式数据库方案，集群通过分片( sharding )来进行数据共享，并提供复制和故障转移功能。
- 节点
  - 一个Redis通常由多个节点组成，向一个节点node发送CLUSTER MEET命令，可以让node节点与ip和port所指定的节点进行握手( handshake )，当握手成功时，node节点就会将ip和port所指定的节点添加到node节点当前所在的集群中。
  - ![image-20220416211232893](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20220416211232893.png)
  - 在集群模式下的节点和普通的单个节点服务器没什么不同，也会该干嘛干嘛，至于那些在集群模式下用到的数据，会保存在响应的数据结构中
  - 数据结构
    - clusterNode 结构保存了一个节点的当前状态，比如节点的创建时间、节点的名字、节点的IP地址和端口号、节点负责哪些槽
    - clusterLink结构保存了连接节点的相关信息，比如套接字描述符、输入输出缓冲区
    - 每个节点都保存着一个clusterstate结构，这个结构记录了在当前节点的视角下，集群目前所处的状态，例如集群是在线还是下线，集群包含多少个节点，集群当前的配置纪元 、还有槽的指派信息
- 槽指派
  - Redis集群通过**分片**的方式来保存数据库中的键值对:集群的整个数据库被分为16384个槽( slot )，数据库中的每个键都属于这16384个槽的其中一个，集群中的每个节点可以处理0个或最多16384个槽。当数据库中的16384个槽都有节点在处理时，集群处于上线状态（ ok );相反地，如果数据库中有任何一个槽没有得到处理，那么集群处于下线状态( fail )。
  - cluster addslots可以向节点指定槽。slots属性是一个二进制位数组( bit array)，这个数组的长度为16384/8=2048个字节
  - 一个节点还会把自己的slots数组信息发送给其他节点，因此集群中每个节点都知道槽分配给了谁
  - 通过将所有槽的指派信息保存在clusterState.slots数组里面，程序要检查槽i是否已经被指派，又或者取得负责处理槽i的节点，只需要访问clusterstate.slots [i]的值即可，这个操作的复杂度仅为O(1)。
  - clusterState.slots数组记录了所有槽的指派信息，而clusterNode.slots只记录了clusterNode 结构所代表的节点的槽指派信息，这是两个slots数组的关键区别所在。
  - cluster addslots执行原理：会把clusterState.slots数组中指向相应节点，clusterNode.slots会把相应的位置1
- 在集群中执行命令
  - ![image-20220416213341590](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20220416213341590.png)
  - 计算键属于哪个槽：CRC16(key)&16383
  - 当节点计算出键所属的槽i之后，节点就会检查自己在clusterstate.slots数组中的项i，判断键所在的槽是否由自己负责，如果clusterstate.slots[i]等于clusterstate.myself，那么说明槽i由当前节点负责，节点可以执行客户端发送的命令。如果clusterstate.slots [ i]不等于clusterstate.myself，那么说明槽i并非由当前节点负责，节点会根据clusterState.slots[ i]指向的clusterNode结构所记录的节点IP和端口号，向客户端返回MOVED错误，指引客户端转向至正在处理槽i的节点。
  - 集群模式下的MOVED信息不会打印出来，单机下会打印
  - 节点和单机服务器在数据库方面的一个区别是，节点只能使用О号数据库，而单机Redis服务器则没有这一限制。
  - 将键值对保存在数据库里面之外，节点还会用clusterstate.slots_to_keys跳跃表来保存槽和键之间的关系，slots_to_keys跳跃表每个节点的分值（ score）都是一个槽号，而每个节点的成员( member)都是一个数据库键
- 重新分片
  - Redis集群的重新分片操作可以将任意数量已经指派给某个节点（源节点）的槽改为指派给另一个节点(目标节点)，并且相关槽所属的键值对也会从源节点被移动到目标节点。实现原理如下：
  - ![image-20220417085952348](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20220417085952348.png)

- ASK错误

  - 在进行重新分片期间，源节点向目标节点迁移一个槽的过程中，可能会出现这样一种情况：属于被迁移槽的一部分键值对保存在源节点里面，而另一部分键值对则保存在目标节点里面。
  - ![image-20220417090413349](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20220417090413349.png)

  - 一个槽里面有多个键
  - `CLUSTER SETSLOT IMPORTING`命令的实现
    - clusterstate结构的importing_slots_from数组记录了当前节点正在从其他节点导入的槽：如果importing slots_from[i]的值不为NULL，而是指向一个clusterNode结构，那么表示当前节点正在从clusterNode所代表的节点导人槽i。
    - 在对集群进行重新分片的时候,向目标节点发送命令：CLUSTER SETSLOT< i >IMPORTING <source_id>，可以将目标节点clusterstate .importing_slots_from[i]的值设置为source_id所代表节点的clusterNode 结构。

  - `CLUSTER SETSLOT MIGRATING`命令的实现
    - clusterstate结构的migrating_slots_to数组记录了当前节点正在迁移至其他节点的槽：如果migrating_slots_to[i]的值不为NULL，而是指向一个clusterNode结构，那么表示当前节点正在将槽i迁移至clusterNode所代表的节点。
    - 在对集群进行重新分片的时候,向源节点发送命令:CLUSTER SETSLOT< i >MIGRATING <target_id>
      可以将源节点clusterstate.migrating_slots_to[i]的值设置为target_id所代表节点的clusterNode 结构。
  - ASKING命令
    - ASKING命令唯一要做的就是打开发送该命令的客户端的REDIS_ASKING标识，当客户端接收到ASK错误并转向至正在导入槽的节点时，客户端会先向节点发送一个ASKING命令，然后才重新发送想要执行的命令，这是因为如果客户端不发送ASKING命令，而直接发送想要执行的命令的话，那么客户端发送的命令将被节点拒绝执行，并返回MOVED错误。
    - 该命令是一个一次性标识
    - ![image-20220417092111011](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20220417092111011.png)

- 复制与故障转移
  - 设置从节点
    - CLUSTER REPLICATE <node_id>：可以让接收命令的节点成为node_id所指定节点的从节点，并开始对主节点进行复制：将自己的clusterstate.myself.slaveof指针指向node_id这个节点、修改clusterstate.myself .flags中的属性、最后调用复制代码
    - 当一个节点成为另一个节点的从节点时，集群中其他的节点也会知道。集群中的所有节点都会在代表主节点的clusterNode结构的slaves属性和numslaves属性中记录正在复制这个主节点的从节点名单。
  - 故障检测
    - 集群中的每个节点都会定期地向集群中的其他节点发送PING消息，以此来检测对方是否在线，如果接收PING消息的节点没有在规定的时间内，向发送PING消息的节点返回PONG消息，那么发送PING消息的节点就会将接收PING消息的节点标记为疑似下线( probable fail，PFAIL)。
    - 修改clusterstate.nodes.clusterNode.fail_reports
    - 半数以上的节点认为是疑似下线，那么就报告是下线
  - 选举新的主节点
    - 基于raft算法进行选举

- 消息
  - 发送消息的类型：MEET消息、PING、PONG、FILE、PUBLISH。消息是由消息头和正文组成