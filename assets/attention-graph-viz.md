```mermaid
graph TD
    Node1[Node 1]
    Node2[Node 2]
    Node3[Node 3]
    Node4[Node 4]
  
    Node1 -->|w1,1| Node1
  
    Node1 -->|w2,1| Node2
    Node2 -->|w2,2| Node2
  
    Node1 -->|w3,1| Node3
    Node2 -->|w3,2| Node3
    Node3 -->|w3,3| Node3
  
    Node1 -->|w4,1| Node4
    Node2 -->|w4,2| Node4
    Node3 -->|w4,3| Node4
    Node4 -->|w4,4| Node4
```