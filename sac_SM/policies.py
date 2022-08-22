import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import numpy as np
from torch.distributions import Normal
import stable_baselines3.common.init as init
import torch.nn.functional as F
from stable_baselines3.sac_SM.distribution import TanhNormal 

 
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, SM_Critic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
    ModularGatedCascadeCondNet,
    
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        env: Union[GymEnv, str],
        embedding_inputs,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        
        # Save arguments to re-create object at loading
        self.env = env
        self.embedding_inputs = embedding_inputs
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        action_dim = get_action_dim(self.action_space)
        '''
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        '''
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        # print('input shape:', self.env.action_space)
        self.FUNC = ModularGuassianGatedCascadeCondContPolicy(input_shape = self.observation_space['obs'].shape[0],
                                                        em_input_shape = np.prod(8,),
                                                        output_shape = 2 * self.action_space.shape[0],
                                                        hidden_shapes = [400,400],
                                                        em_hidden_shapes = [400],
                                                        num_layers = 2,
                                                        num_modules = 2,
                                                        module_hidden = 256,
                                                        num_gating_layers = 2,
                                                        gating_hidden = 256,
                                                        add_bn = False,
                                                        pre_softmax = False,
                                                        base_type = MLPBase,
                                                        )
        

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            
            
            
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)
        
        
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor, embedding_input) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        # change to soft modularization
        obs = obs['obs']
        mean_actions, std, log_std = self.FUNC.forward(obs, embedding_input, return_weights = False)
        '''
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        
        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        '''
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, obs['task'])
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        embedding_input = obs['task']
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, embedding_input)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)

class EmbeddingGuassianContPolicyBase:

    def eval_act( self, x, embedding_input ):
        with th.no_grad():
            mean, std, log_std = self.forward(x, embedding_input)
        return th.tanh(mean.squeeze(0)).detach().cuda().numpy()
    
    def explore( self, x, embedding_input, return_log_probs = False, return_pre_tanh = False ):
        
        mean, std, log_std = self.forward(x, embedding_input)

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True) 
        
        dic = {
            "mean": mean,
            "log_std": log_std,
            "ent":ent
        }

        if return_log_probs:
            action, z = dis.rsample( return_pretanh_value = True )
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample( return_pretanh_value = True )
                dic["pre_tanh"] = z.squeeze(0)
            action = dis.rsample( return_pretanh_value = False )

        dic["action"] = action.squeeze(0)
        return dic

    def update(self, obs, embedding_input, actions):
        mean, std, log_std = self.forward(obs, embedding_input)
        dis = TanhNormal(mean, std)

        log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
        ent = dis.entropy().sum(1, keepdim=True) 
        
        out = {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "ent": ent
        }
        return out

class ModularGuassianGatedCascadeCondContPolicy(ModularGatedCascadeCondNet, EmbeddingGuassianContPolicyBase):
    def forward(self, obs: th.Tensor, embedding_input, return_weights = False ):
        obs = super().forward(obs, embedding_input, return_weights = return_weights)
        if isinstance(obs, tuple):
            general_weights = obs[1]
            last_weights = obs[2]
            obs = obs[0]

        mean, log_std = obs.chunk(2, dim=-1)

        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = th.exp(log_std)

        if return_weights:
            return mean, std, log_std, general_weights, last_weights
            # return mean, std, log_std, general_weights
        return mean, std, log_std

    def eval_act( self, x: th.Tensor, embedding_input, return_weights = False ):
        with th.no_grad():
            if return_weights:
                # mean, std, log_std, general_weights, last_weights = self.forward(x, embedding_input, return_weights)
                mean, std, log_std, general_weights = self.forward(x, embedding_input, return_weights)
            else:
                mean, std, log_std = self.forward(x, embedding_input, return_weights)
        if return_weights:
            # return torch.tanh(mean.squeeze(0)).detach().cpu().numpy(), general_weights, last_weights
            return th.tanh(mean.squeeze(0)).detach().cuda().numpy(), general_weights
        return th.tanh(mean.squeeze(0)).detach().cuda().numpy()


    def explore( self, x: th.Tensor, embedding_input, return_log_probs = False,
                    return_pre_tanh = False, return_weights = False ): # x is observation.
        if return_weights:
            mean, std, log_std,  general_weights, last_weights = self.forward(x, embedding_input, return_weights)
            # general_weights, last_weights = weights
            dic = {
                "general_weights": general_weights,
                "last_weights": last_weights
            }
        else:
            mean, std, log_std = self.forward(x, embedding_input)
            dic = {}

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True) 
        
        dic.update({
            "mean": mean,
            "log_std": log_std,
            "ent":ent
        })

        if return_log_probs:
            action, z = dis.rsample( return_pretanh_value = True )
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample( return_pretanh_value = True )
                dic["pre_tanh"] = z.squeeze(0)
            action = dis.rsample( return_pretanh_value = False )

        dic["action"] = action.squeeze(0)
        return dic
 
  
class MLPBase(nn.Module):
    def __init__(self, input_shape, hidden_shapes, activation_func=F.relu, init_func = init.basic_init, last_activation_func = None ):
        super().__init__()
        
        self.activation_func = activation_func
        self.fcs = []
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        input_shape = np.prod(input_shape)

        self.output_shape = input_shape
        for i, next_shape in enumerate( hidden_shapes ):
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("fc{}".format(i), fc)

            input_shape = next_shape
            self.output_shape = next_shape

    def forward(self, x):
        
        out = th.Tensor(x).cuda()
        for fc in self.fcs[:-1]:
            out = fc(out)
            out = self.activation_func(out)
        out = self.fcs[-1](out)
        out = self.last_activation_func(out)
        return out


class SACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        env: Union[GymEnv, str],
        embedding_inputs,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        self.env = env
        self.embedding_inputs = embedding_inputs
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs, env = self.env, embedding_inputs = self.embedding_inputs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SM_Critic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return SM_Critic(**critic_kwargs, env = self.env, embedding_inputs = self.embedding_inputs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = SACPolicy


class CnnPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class MultiInputPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        env: Union[GymEnv, str],
        embedding_inputs,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            env,
            embedding_inputs,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
