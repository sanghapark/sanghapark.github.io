# Deep Learning in Knowledge Graph

### Abstract

Three categories of deep learning-based KG techniques:

1. **knowledge representation learning** techniques which embed entities and relations in a KG into a dense, low-dimensional, and real-valued semantic space
2. **neural relation extraction** techniques which extract facts/relations from text, which can then be used to construct/complete KG
3. **deep learning-based entitiy linking** techniques which bridge KG with textual data, which can facilitate many different taks.



### 1. Knowledge Representaiton Learning

Representation learning aims at embedding the objects into a dense, low-dimensional, and real-valued semantic space. Knowledge representation learning is a subarea of representation learning, which focuses on embedding the entities and relations in KG.

Recent studies reveal that **translation-based representation learning** methods are efficient and effective to encode relational facts in KG with low-dimensional representations of both entities and relations, which can alleviate the issue of data sparsity and be further employed to knowledge acquisition, fusion, and inference. 

TransE is one typical translation-base knowledge representation learning methods. TransE regards the relation in a relational triple as a translation between the embeddings of the head and tail entities, that is, 
$$
\vec{\text{h}} + \vec{\text{r}} \approx \vec{\text t}
$$
where the triple $$(h, r, t)$$ holds.

Although TransE has achieved great success. it still has issues when modeling 1-to-N, N-to-1, and N-to-N relations. The entity embeddings learnt by TransE are lacking in discrimination due to these complex relations. **How to deal with complex relations is one of the key challenges in knowledge representaiton learning.**

TransH and TransR are proposed to represent an entity with different representations when involved in different relations.

- **TransH** models the relation as a translation vector on a hyperplane and projects the entity embeddings into the hyperplane with a normal vector. 

- **TransR** represents entities in the entity semantic space and uses a relation specific transform matrix to project it into the different relation spaces when involved in different relations. 

- **TransD** considers the information of entities in the projecting matrices

- **TransSparse** which considers the heterogeneity and imbalance of relations via sparse matrices.

- **TrangG** focus on different characteristics of relations

- **KG2E** adopt Gaussian embeddings to model both entities and relations.

- **ManifoldE** employs a manifold-based embedding principle

  

TransE still has a problem that only considering direct relations between entities. 

- **Path-based TransE** which extends TransE to model relational paths by selecting reasonable relational paths and representing them with low-dimensional vectors.

Most existing knowledge representation learning methods discussed above only focus on the structure information in KG, regardless of the rich multisource information such textual information, type information, and visual information.  These cross-modal information can provide supplementary knowledge of the entities specially for those entities with less relational facts and is significant when learning knowledge representations. 

For textual information, Wang and Zhong ([*Knowledge Graph and Text Jointly Embedding* 2014](https://www.aclweb.org/anthology/D14-1167)) propose to jointly embed both entities and words into a unified semantic space by aligning them with entity names, descriptions, and Wikipedia anchors. 6

​	

### 2. Neural Relation Extraction



































