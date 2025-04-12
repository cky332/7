import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mi_estimators import CLUBSample

# python milk_main.py --dataset baby --reg_coeff 1e-3 --penalty_coeff 100   --lr 1e-3  --exp_mode fm
# python milk_main.py --dataset baby --reg_coeff 1e-3 --penalty_coeff 300   --lr 1e-3  --exp_mode fm
# python milk_main.py --dataset baby --reg_coeff 1e-3 --penalty_coeff 700   --lr 1e-3  --exp_mode fm

class MGCN(torch.nn.Module):
    def __init__(self, edge_index, num_user, num_item, dim_feat, dim_latent):
        super(MGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.n_layers = 3

        # self.item_id_preference = torch.nn.Embedding(num_embeddings=num_item, embedding_dim=self.dim_latent)
        # torch.nn.init.normal_(self.item_id_preference.weight, std=0.1)

        self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
        
    def forward(self, features, user_id_preference):
        
        # temp_features = torch.mul(self.item_id_preference.weight, self.MLP(features)) 
        temp_features = self.MLP(features)
        all_emb = torch.cat((user_id_preference, temp_features),dim=0)
        all_emb = F.normalize(all_emb)
   
        embs = [all_emb]
        g_droped = self.edge_index    
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_user, self.num_item])
        
        return users, items



class MILK_model(torch.nn.Module):
    def __init__(self, env, dataset):
        super(MILK_model, self).__init__()
        self.env = env
        self.n_layers = 3
        self.A_split = False
        self.n_user = dataset.n_user
        self.m_item = dataset.m_item
        self.Graph = dataset.getSparseGraph()
        self.free_emb_dimension = self.env.args.free_emb_dimension  # free embedding的维数

        self.audio_feat = None


        self.image_feat = torch.tensor(dataset.image_feat, dtype=torch.float32).to(self.env.device)
        self.image_feat = torch.nn.functional.normalize(self.image_feat)
        self.v_gcn = MGCN(self.Graph, self.n_user, self.m_item, self.image_feat.size(1), self.free_emb_dimension)


        self.text_feat = torch.tensor(dataset.text_feat, dtype=torch.float32).to(self.env.device)
        self.text_feat = torch.nn.functional.normalize(self.text_feat)
        self.t_gcn = MGCN(self.Graph, self.n_user, self.m_item, self.text_feat.size(1), self.free_emb_dimension)

        if self.env.args.dataset == 'tiktok':
            self.audio_feat = torch.tensor(dataset.audio_feat, dtype=torch.float32).to(self.env.device)
            self.audio_feat = torch.nn.functional.normalize(self.audio_feat)
            self.a_gcn = MGCN(self.Graph, self.n_user, self.m_item, self.audio_feat.size(1), self.free_emb_dimension)


        self.user_emb = torch.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=self.free_emb_dimension)
        self.item_emb = torch.nn.Embedding(
            num_embeddings=self.m_item, embedding_dim=self.free_emb_dimension)
        
        self.fusion_linear = torch.nn.Linear(self.free_emb_dimension, self.free_emb_dimension, bias=False)

        self.fusion_linear = nn.Sequential(
            nn.Linear(self.free_emb_dimension, self.free_emb_dimension, bias=False),
            nn.Dropout(),
            nn.Tanh()
        )

        self.init_mi_estimator()

        self.final_item = None
        self.final_user = None
        self.activate = torch.nn.Sigmoid()

        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        torch.nn.init.normal_(self.item_emb.weight, std=0.1)
        torch.nn.init.eye_(self.fusion_linear[0].weight)  # 初始化为单位矩阵
        self.to(self.env.device)


    def init_mi_estimator(self) :
        self.item_image_estimator = CLUBSample(self.image_feat.size(1), self.free_emb_dimension, 64).cuda()
        self.item_text_estimator = CLUBSample(self.text_feat.size(1), self.free_emb_dimension, 64).cuda()
        # self.item_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()

        params = list(self.item_image_estimator.parameters()) + list(self.item_text_estimator.parameters())

        self.optimizer_club = torch.optim.Adam(params, lr = 1e-4)
        self.scheduler_club = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_club, 'min')

    def pre_epoch_processing(self) :

        item_image_g = self.image_feat
        item_text_g = self.text_feat

        with torch.no_grad():
            item_image_s = self.v_gcn.MLP(self.image_feat)
            item_text_s = self.t_gcn.MLP(self.text_feat)


        total_loss_mi = 0.0
        for _ in range(10) :
            self.item_image_estimator.train(); self.item_text_estimator.train()

            item_rand_idx = torch.randperm(self.m_item)[:2048]

            loss_mi = 0.0
        
            loss_mi += self.item_image_estimator.learning_loss(item_image_g[item_rand_idx], item_image_s[item_rand_idx])
            loss_mi += self.item_text_estimator.learning_loss(item_text_g[item_rand_idx], item_text_s[item_rand_idx])

            self.optimizer_club.zero_grad()
            loss_mi.backward(retain_graph = True)
            self.optimizer_club.step()
            total_loss_mi += loss_mi.detach().item()

        self.scheduler_club.step(total_loss_mi)
        self.item_image_estimator.eval(); self.item_text_estimator.eval()
        print(total_loss_mi)

        # loss_club = 0.0
        # loss_club += self.item_image_estimator(mask_item_embs[0],item_image_g[batch_pos_items])
        # loss_club += self.item_text_estimator(mask_item_embs[0],item_text_g[batch_pos_items])



        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        # if user_emb.grad is not None:
        #     print(user_emb.grad.mean())
        #     print(item_emb.grad.mean())
        #     assert torch.isnan(user_emb).sum() == 0
        #     assert torch.isnan(item_emb).sum() == 0

        all_emb = torch.cat([user_emb, item_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_user, self.m_item])
        return users, items



    def forward(self, random=False):
        """
        propagate methods for lightGCN
        """
        
        user_id_emb = self.user_emb.weight


        v_user_emb, v_item_emb = self.v_gcn(self.image_feat, user_id_emb)
        t_user_emb, t_item_emb = self.t_gcn(self.text_feat, user_id_emb)

        if self.audio_feat is None:
            user_emb = user_id_emb + (v_user_emb + t_user_emb) / 2 
            item_emb = self.fusion_linear((v_item_emb.detach() + t_item_emb.detach()) / 2)


        else:
            a_user_emb, a_item_emb = self.a_gcn(self.audio_feat, user_id_emb)

            user_emb = user_id_emb + (v_user_emb + t_user_emb + a_user_emb) / 3 
            item_emb = self.fusion_linear((v_item_emb.detach() + t_item_emb.detach() + a_item_emb.detach()) / 2)

 
        # print(self.fusion_linear.weight)

        assert torch.isnan(user_emb).sum() == 0
        assert torch.isnan(item_emb).sum() == 0
        self.final_user = user_emb
        self.final_item = item_emb

        return user_emb, item_emb

    def get_env_emb(self, mix_ration, env):
        """
        propagate methods for lightGCN
        """

        user_id_emb = self.user_emb.weight

        v_user_emb, v_item_emb = self.v_gcn(self.image_feat, user_id_emb)

        t_user_emb, t_item_emb = self.t_gcn(self.text_feat, user_id_emb)

        v_mm_emb = mix_ration[env][0] *  v_item_emb 
        t_mm_emb = mix_ration[env][1] *  t_item_emb 

        if self.audio_feat is None:
            item_emb = torch.cat([v_mm_emb, t_mm_emb], dim=1)
            user_emb = torch.cat([user_id_emb + v_user_emb, user_id_emb + t_user_emb], dim=1)

        else:
            a_user_emb, a_item_emb = self.a_gcn(self.audio_feat, user_id_emb)
            a_mm_emb = mix_ration[env][2] *  a_item_emb 
            
            item_emb = torch.cat([v_mm_emb, t_mm_emb, a_mm_emb], dim=1)
            user_emb = torch.cat([user_id_emb + v_user_emb, user_id_emb + t_user_emb, user_id_emb + a_user_emb], dim=1)
        

        assert torch.isnan(user_emb).sum() == 0
        assert torch.isnan(item_emb).sum() == 0

        return user_emb, item_emb



    def invariant_learning_emb(self, batch_users, batch_pos_items, batch_neg_items):
        
        modality_mask = [[1,1], [1, 0], [0, 1]]
        max_mutual_loss = 0.
 
        user_id_emb = self.user_emb.weight

        v_user_emb, v_item_emb = self.v_gcn(self.image_feat, user_id_emb)
        t_user_emb, t_item_emb = self.t_gcn(self.text_feat, user_id_emb)


        if self.audio_feat is None:
            user_emb = user_id_emb + (v_user_emb + t_user_emb) / 2 
            v_item_detach_emb = v_item_emb.detach()
            t_item_detach_emb = t_item_emb.detach()

        else:
            a_user_emb, a_item_emb = self.a_gcn(self.audio_feat, user_id_emb)

            user_emb = user_id_emb + (v_user_emb + t_user_emb + a_user_emb) / 3 

            v_item_detach_emb = v_item_emb.detach()
            t_item_detach_emb = t_item_emb.detach()
            a_item_detach_emb = a_item_emb.detach()


        mask_item_embs = []
        for mask_indication in modality_mask:

            mask_fusion_emb = (mask_indication[0] * v_item_detach_emb + mask_indication[1] * t_item_detach_emb) / 2

            mask_item_emb = self.fusion_linear(mask_fusion_emb)
            mask_item_embs.append(mask_item_emb[batch_pos_items])

        mask_item_embs = torch.stack(mask_item_embs, dim=0)
        # print(mask_item_embs.shape)

        for i in range(len(modality_mask)):
            for j in range(len(modality_mask)):
                if i != j:
                    # print(mask_item_embs[i].shape, mask_item_embs[j].shape)
                    max_mutual_loss += self.InfoNCE(mask_item_embs[i], mask_item_embs[j])

        max_mutual_loss += max_mutual_loss / 6

        # 互信息的最小化项

        item_emb = self.fusion_linear((v_item_emb.detach() + t_item_emb.detach()) / 2)

        item_image_g = self.image_feat
        item_text_g = self.text_feat
        loss_club = 0.0
        loss_club += self.item_image_estimator(item_image_g[batch_pos_items], item_emb[batch_pos_items])
        loss_club += self.item_text_estimator(item_text_g[batch_pos_items], item_emb[batch_pos_items])

        min_mutual_loss = (loss_club/2)

        # item_emb_cl_loss = self.InfoNCE(item_emb[batch_pos_items], v_item_emb[batch_pos_items].detach()) + self.InfoNCE(item_emb[batch_pos_items], t_item_emb[batch_pos_items].detach())

        # max_mutual_loss = (item_emb_cl_loss/2)

        all_user_emb, all_item_emb =  self.forward()

        user_emb  = all_user_emb[batch_users]
        pos_items = all_item_emb[batch_pos_items]
        neg_items = all_item_emb[batch_neg_items]

        pos_scores = torch.sum(torch.mul(user_emb, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_items), dim=1)

        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        return bpr_loss, max_mutual_loss, min_mutual_loss

    def InfoNCE(self, view1, view2, temperature = 0.4):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)


    def calculate_reg_loss(self, batch_users, batch_pos_items, batch_neg_items):

        all_user_emb, all_item_emb =  self.forward()
        # all_user_emb = self.user_emb.weight
        # all_item_emb = self.item_emb.weight

        reg_embedding_loss = (1 / 2) * (all_user_emb[batch_users].norm(2).pow(2) + all_item_emb[batch_pos_items].norm(2).pow(2) + all_item_emb[batch_neg_items].norm(2).pow(2)) / float(len(batch_users))

        # reg_model_loss = (1 / 2) * self.fusion_linear.weight.norm(2).pow(2) * 1e-4

        return reg_embedding_loss