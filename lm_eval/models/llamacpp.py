#=====================================================================#
#=====================================================================#
import os
import logging
import time
import requests
import transformers
from requests.exceptions import RequestException
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import Any, Dict, List, Optional, Tuple, Union
import subprocess
from tqdm import tqdm
from llama_cpp import Llama
from jinja2 import Template, meta
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, TemplateLM
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    handle_stop_sequences,
    pad_and_concat,
    stop_sequences_criteria,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)
logger = logging.getLogger(__name__)
#=====================================================================#
#=====================================================================#

#-------------------------------------#

def get_result(
        logprobs:Dict[str,Any],
        context_length:int
        ) -> Tuple[float, bool]:

    """
    logprobsの情報から、context 長さ以降の継続部分の log probabilityと greedy 生成か否かの判定結果を返します。
    """
    is_greedy: bool = True
    offsets: List[int] = logprobs["text_offset"]
    tokens: List[str] = logprobs["tokens"]
    tokens_logprobs: List[float] = logprobs["token_logprobs"]

    idx: int = 0
    while idx < len(offsets) and offsets[idx] < context_length:
        idx += 1
    continuation_logprobs: float = sum(tokens_logprobs[idx:-1])
    for i in range(idx, len(tokens)):
        token: str = tokens[i]
        top_tokens: Dict[str, float] = logprobs["top_logprobs"][i]
        top_token: str = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy
#-------------------------------------#

#=====================================================================#
#=====================================================================#

@register_model("llama-cpp", "llamacpp")
class LlamaCppLM(LM):

    def __init__(
            self,            
            path_to_model:str,
            path_to_output:str,
            quantization_format:str = 'Q4_K_M',
            path_to_conversion_script:str='/Users/tsutomunagayoshi/research/llm/llama.cpp/convert_hf_to_gguf.py',
            path_to_quantize_script:str= '/Users/tsutomunagayoshi/research/llm/llama.cpp/build/bin/llama-quantize',
            logprobs:Optional[int]=10,
            batch_size:int=1, # batch size for generate_until and loglikelihood
            n_gpu_layer:int=0,
            n_ctx:int=2048, # size of context window
            max_tokens:int=512, # the maximum size of generation token size 
            last_n_tokens_size:int=64, # Maximum number of tokens to keep in the last_n_tokens deque
            n_batch=512, # Prompt processing maximum batch size for Llama
            temperature:float=0.00,
            repeat_penalty:float=1.2,
            verbose:bool=False,
            **kwargs,
            )->None:
        super().__init__()

        '''
        Set parameters
        '''

        self.path_to_model      = path_to_model       
        self.path_to_output     = path_to_output     
        self.quantization_format= quantization_format
        self.n_gpu_layer        = n_gpu_layer       
        self.n_ctx              = n_ctx                
        self.n_batch            = n_batch             
        self.batch_size         = batch_size
        self.last_n_tokens_size = last_n_tokens_size
        self.logprobs           = logprobs
        self.max_tokens         = max_tokens           
        self.temperature        = temperature
        self.repeat_penalty     = repeat_penalty        
        self.verbose            = verbose            

        self.path_to_conversion_script = path_to_conversion_script
        self.path_to_quantize_script   = path_to_quantize_script

        if torch.backends.mps.is_available() and torch.backends.mps.is_built() :
            self.device = 'mps'
        else :
            self.device = 'cpu'

        '''
        scheduler
        '''
        # self.batch_sizes = {}
        # self.batch_schedule = 1
        # self.max_batch_size = 1

        # if str(batch_size).startswith("auto"):
        #     batch_size = batch_size.split(":")
        #     self.batch_size_per_gpu = batch_size[0]
        #     self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        # else:
        #     self.batch_size_per_gpu = int(batch_size)


        '''
        Quantization 
        '''

        self.path_to_gguf = os.path.join(
            self.path_to_output,
            '{}_{}.gguf'.format(os.path.basename(self.path_to_model),self.quantization_format)
        )
        self.__convert2gguf()

        assert os.path.exists(self.path_to_gguf), {
            '----> {} does not exist ! '.format(self.path_to_gguf)
        }

        '''
        Set model
        '''
        self.model = Llama(
            model_path=self.path_to_gguf,
            n_gpu_layers=self.n_gpu_layer,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=None,
            n_threads_batch=None,
            logits_all=True,
            embedding=False,
            last_n_tokens_size=self.last_n_tokens_size,
            verbose=self.verbose,
            type_k=None,
            type_v=None,
        )

        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     path_to_model,
        # )
        # self.prefix_token_id = self.tokenizer.eos_token_id
        # print ('===============\nprefix token id: {}'.format(self.prefix_token_id))
        # exit()
    #---------------#

    def __convert2gguf(self)->None :

        '''
        Convert to just gguf
        '''

        if os.path.exists(self.path_to_gguf) :
            return

        path_to_gguf_tmp = os.path.join(
            self.path_to_output,
            '{}_noquantization.gguf'.format(os.path.basename(self.path_to_model))
        )

        try :
            result = subprocess.run(
                [
                    'python',
                    self.path_to_conversion_script,
                    self.path_to_model,
                    '--outfile',
                    path_to_gguf_tmp,
                    '--outtype',
                    'f16',
                ]
            )
            if result.returncode == 0 :
                print (f'Success in converting ! : {path_to_gguf_tmp}')
                print (f'result.stdout: {result.stdout}')
            else :
                print ('Failure in converting ...')
                print (f'result.stderr: {result.stderr}')

        except Exception as e :
            print (f'Error to convert to no quantized gguf !: {e}')


        '''
        Convert the simple gguf to quantized gguf
        '''
        try :
            result = subprocess.run(
                [
                    self.path_to_quantize_script,
                    path_to_gguf_tmp,
                    self.path_to_gguf,
                    self.quantization_format,
                ]
            )
            if result.returncode == 0 :
                print (f'Success in converting ! : {self.path_to_gguf}')
                print (f'result.stdout: {result.stdout}')
            else :
                print ('Failure in converting ...')
                print (f'result.stderr: {result.stderr}')

        except Exception as e :
            print (f'Error to quantize gguf !: {e}')

    #---------------#

    def _llama_completion(self,
                          prompt:str,
                          max_tokens:int,
                        #   temperature:float=0.7,
                        #   repeat_penalty:float=1.2,
                          stop:Optional[Union[List[str],str]]=None,
                          echo:bool=False,
                          )->Dict[str,Any] :

        """
        llamacpp のモデル呼び出しをラップするヘルパー関数

        Args:
            prompt (str): 入力プロンプト
            stop (list or None): 生成停止文字列のリスト
            echo (bool): 入力も出力に含めるかどうか
            echo パラメータは、モデルに対してプロンプト自体も出力に含めるかどうかを指示するものです。
	            •	echo=False（デフォルト）
	            •	プロンプトは出力結果に含まれず、生成されたテキストのみが返されます。
	            •	echo=True
	            •	プロンプトとその後に生成されたテキストの両方が返されます。
	            •	これにより、入力されたプロンプトと生成された続きの部分を合わせた全文を確認することができます。
                この機能は、例えば評価時にプロンプトと生成結果の両方を表示したい場合や、生成されたテキストの前提となるプロンプトを明示的に確認したい場合に有用です。
        Returns:
            dict: モデルからの出力
        """
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
            "logprobs":self.logprobs,
            "echo": echo
        }
        if stop is not None:
            kwargs["stop"] = stop
        # print ('-----------\n')
        # print ('PROMPT in llama completion : {}'.format(self.get_token_size(prompt)))
        if self.get_token_size(prompt)>self.n_ctx :

            dummy_response = {
                "choices": [
                    {
                        "text": "[DUMMY OUTPUT: INPUT TOO LARGE]",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length"
                    }
                ],
                "usage": {
                    "prompt_tokens": self.get_token_size(prompt),
                    "completion_tokens": 0,
                    "total_tokens": self.get_token_size(prompt),
                }
            }
            return dummy_response  # ダミーのレスポンスを返す
        else :
            response: Dict[str, Any] = self.model(prompt, **kwargs)
            # print ('RESPONSE in llama completion: {}'.format(self.get_token_size(response['choices'][0]['text'])))
            # from pprint import pprint
            # pprint (response)
            # exit()
            return response
    #---------------#

    def _loglikelihood_tokens(
        self,
        model_hf:PreTrainedModel,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
        is_verbose:bool=False,
        ) -> List[Tuple[float, bool]]:


        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc='Running loglikelihood requests',
        )

        results = []

        for _, context_enc, continuation_enc in requests :

            '''
            Preprocessing for logit calculation
            '''
            if is_verbose :
                print ('\n\n\n--->')
                print ('context_enc: {}'.format(context_enc))
                print ('continuation_enc: {}'.format(continuation_enc))

            padding_len_inp = None
            inp = torch.tensor(
                (context_enc + continuation_enc)[-(self.n_ctx+1):][:-1],
                dtype=torch.long,
                device=self.device,
            )
            (inplen,) = inp.shape

            padding_len_inp = (max(padding_len_inp,inplen) if padding_len_inp is not None else inplen)

            batched_inps = pad_and_concat(padding_len_inp, [inp], padding_side='right') # should be torch.size([batch,paddig_len_inp])
            # print ('batched inputs : {}'.format(batched_inps.size()))    

            if is_verbose :
                print ('context_enc: {}'.format(len(context_enc)))
                print ('continuation_enc: {}'.format(len(continuation_enc)))
                print ('input length: {}'.format(inplen))
                print ('padding length of input: {}'.format(padding_len_inp))
                print('batched inputs: {} ({})'.format(batched_inps.size(),type(batched_inps)))

            '''
            Get Logits
            '''

            multi_logits = F.log_softmax(
                    model_hf(batched_inps).logits,
                    dim=-1
                )  # [batch,padding_length] ---> [batch, padding_length (inp or cont), vocab] ---> log softmax ---> [batch, padding_length (inp or cont), vocab] 
            # print ('multi_logits: {} ({})'.format(multi_logits.size(),type(multi_logits)))

            # input_tokens = list(batched_inps.cpu().numpy()[0]) # list[int]
            # print (type(input_tokens), len(input_tokens))

            if is_verbose :
                print ('multi logits: {} ({})'.format(multi_logits.size(),type(multi_logits)))

            '''
            Calculate loglikelihood and greedy
            '''
            continuation_len = len(continuation_enc)
            context_len = inplen + (multi_logits[0].shape[0] - padding_len_inp)
            # logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len) # torch.Size([length of continuation, vocab])
            logits = multi_logits[0][context_len - continuation_len: inplen]
            logits = logits.unsqueeze(0)
            # print ('continuation length: {}'.format(continuation_len))
            # print ('context length: {}'.format(context_len))


            greedy_tokens = logits.argmax(dim=-1)
            continuation_tokens = torch.tensor(
                continuation_enc, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            max_equal = (greedy_tokens == continuation_tokens).all()
            # print ('continuation_tokens: {} ({})'.format(continuation_tokens.size(),type(continuation_tokens)))

            logits = torch.gather(logits, 2, continuation_tokens.unsqueeze(-1)).squeeze(-1)
            answer = (float(logits.sum()), bool(max_equal))
            results.append(answer)

            pbar.update(1)

        pbar.close()
        return results
    #---------------#

    def _encode_pair(self, context:str, continuation:str)->tuple[List[int],List[int]] :
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0 :
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context+continuation)
        context_enc = self.tok_encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return (context_enc,continuation_enc)        
    #---------------#

    def tok_encode(self, string:str, left_truncate_len=None, add_special_tokens=None)->List[int] :
        return self.model.tokenize(string.encode('utf-8'), add_bos=True, special=True)
    #---------------#

    def loglikelihood(
        self,
        requests:List[Instance],
        disable_tqdm:bool=False,
        ) -> List[Tuple[float, bool]]:

        '''
        Since HF is much faster than llama-cpp-python to perform logit calculation even in your poor M2 chip (14.12 sec vs 1.40 sec), 
        you need load model twice.
        '''

        requests_new = []

        for context, continuation in [req.args for req in requests]:
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    # [self.prefix_token_id],
                    [self.model.token_eos()],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            requests_new.append(((context,continuation),
                                 context_enc,
                                 continuation_enc
                                 ))

        model_hf:PreTrainedModel = AutoModelForCausalLM.from_pretrained(self.path_to_model).to(self.device)
        return self._loglikelihood_tokens(
            model_hf=model_hf,
            requests=requests_new,
            disable_tqdm=disable_tqdm,
            )
    #---------------#

    def loglikelihood_rolling(
            self,
            requests:List[Instance],
            disable_tqdm:bool=False,
            ) -> Any:
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )        
    #---------------#

    def generate_until(self,
                       requests:list[Instance],
                       disable_tqdm:bool=False,
                       ) -> list[str]:
        '''
        Each request contains Instance.args : 
        - Tuple[str, dict] containing 
            - 1. an input string to the LM and 
            - 2. a dictionary of keyword arguments used to control generation parameters.
        - Using this input and these generation parameters,
          text will be sampled from the language model 
          (typically until a maximum output length or specific stopping string sequences
          --for example, {"until": ["\n\n", "."], "max_gen_toks": 128}).
        - The generated output text from the model will then be returned.
        '''

        if not requests :
            return []
        
        outputs:List[Optional[str]] = []
        for req in tqdm([req.args for req in requests],disable=disable_tqdm) :

            inp:str = req[0]
            request_args:Dict[str,Any] = req[1]

            until:Union[List[str],str] = request_args.get('until',['</s>'])
            max_gen_toks:int = request_args.get('max_gen_toks',self.max_tokens)

            response:Dict[str,Any] = self._llama_completion(
                prompt=inp,
                max_tokens=max_gen_toks,
                # temperature=request_args.get('temperature',self.temperature),
                # repeat_penalty=request_args.get('repeat_penalty',self.repeat_penalty),
                stop=until,
                echo=False,
            )

            if response and 'choices' in response and response['choices'] :
                choice:Dict[str,Any] = response['choices'][0]
                if "text" in choice:
                    generated_text: str = choice["text"].strip()
                    outputs.append(generated_text)
                else:
                    logger.error(f"Invalid response for generate_until. Missing 'text'. Response: {response}")
                    outputs.append(None)
            else:
                logger.error(f"Invalid response for generate_until. Response: {response}")
                outputs.append(None)

        return outputs
    #---------------#

    def get_token_size(self, text: str) -> int:
        """
        文字列を入力として、そのトークンサイズ（トークン数）を取得する。

        Args:
            text (str): トークン化する文字列

        Returns:
            int: トークン数
        """
        tokens = self.model.tokenize(text.encode("utf-8"), add_bos=False)
        return len(tokens)    
    #---------------#

    # @property
    # def tokenizer_name(self) -> str:
    #     """
    #     キャッシュ用に使用される tokenizer 名を返します。
    #     """
    #     return "llamacpp"
    #---------------#


    #---------------#

#=====================================================================#
#=====================================================================#

'''
生成に要求されるトークン数は、基本的に次の2つの要素の合計で決まります。
	1.	入力プロンプトのトークン数
	•	すでに入力されているプロンプト（コンテキスト）をトークナイズしたときのトークン数です。
	2.	生成するために要求したトークン数
	•	生成を制御するパラメータ（たとえば max_tokens や max_gen_toks）で指定された、新たに生成するトークンの上限の値です。

つまり、「プロンプトのトークン数 + max_tokens」 の合計が、モデルのコンテキストウィンドウ（例えば 2048 トークン）を超えないようにしなければなりません。たとえば、プロンプトが1500トークンで、max_tokens を2000と指定すると、合計は3500トークンとなり、コンテキストウィンドウの上限2048を大幅に超えてしまいます。

⸻

コンテキストウィンドウを超えた場合の対応策

直接、ライブラリ内部で「ウィンドウを超えた時点で生成を途中で自動停止する」という設定が用意されているケースは一般的には少ないです。ほとんどの場合、生成前にmax_tokens の値を適切に設定することで、生成されるトークン数の合計がコンテキストウィンドウ内に収まるようにします。

具体的には、以下の方法が考えられます。
	•	事前チェックによる自動調整
プロンプトをトークナイズして、そのトークン数を計算し、残りの分（例：context_window_size - prompt_token_count）を max_tokens として設定するロジックを組む。
これにより、生成要求がコンテキストウィンドウを超えないように制限できます。
	•	静的なパラメータ設定
あらかじめプロンプトの長さが一定の範囲内であると分かっている場合、その長さを考慮して max_tokens の値を設定する。たとえば、プロンプトが常に1000トークン程度なら、max_tokens を 1024 にするなど。

⸻

まとめると、生成に要求されるトークン数は**「プロンプトのトークン数」と「max_tokens で指定した生成トークン数」の合計**で決まり、この合計がコンテキストウィンドウを超えるとエラーになります。
直接「超えた時点で生成を中断する」ような自動機能は通常は用意されていないため、事前に max_tokens の値を適切に設定して、常にコンテキストウィンドウ内に収まるようにする必要があります。
'''


'''
{'choices': [{'finish_reason': 'stop',
              'index': 0,
              'logprobs': {'text_offset': [1603, 1604, 1606, 1607, 1609],
                           'token_logprobs': [np.float32(-0.0039992128),
                                              np.float32(-0.00041071087),
                                              np.float32(-0.0001315984),
                                              np.float32(-0.08028601),
                                              np.float32(-0.016384935)],
                           'tokens': ['コ', 'ック', 'リ', 'さん', '\n\n'],
                           'top_logprobs': [{'ア': np.float32(-9.998671),
                                             'アン': np.float32(-8.832893),
                                             'ウ': np.float32(-6.0755363),
                                             'ウィ': np.float32(-8.674629),
                                             'オ': np.float32(-8.433865),
                                             'ク': np.float32(-10.102403),
                                             'コ': np.float32(-0.0039992128),
                                             'サ': np.float32(-9.973893),
                                             'シ': np.float32(-10.157959),
                                             'ハ': np.float32(-9.860619)},
                                            {'ock': np.float32(-11.212922),
                                             'イン': np.float32(-10.348887),
                                             'ク': np.float32(-10.63721),
                                             'コ': np.float32(-10.924699),
                                             'ッ': np.float32(-9.084537),
                                             'ック': np.float32(-0.00041071087),
                                             'ックス': np.float32(-9.960038),
                                             'ッチ': np.float32(-10.587617),
                                             'ド': np.float32(-10.224547),
                                             'ーン': np.float32(-10.798329)},
                                            {'RI': np.float32(-14.3183365),
                                             'り': np.float32(-11.906313),
                                             'ィ': np.float32(-13.517651),
                                             'トリ': np.float32(-13.948828),
                                             'ビ': np.float32(-14.28101),
                                             'リ': np.float32(-0.0001315984),
                                             'リン': np.float32(-12.753296),
                                             'リー': np.float32(-9.16736),
                                             'ル': np.float32(-11.851406),
                                             'レ': np.float32(-13.842673)},
                                            {'': np.float32(-7.7200813),
                                             '\n': np.float32(-5.861108),
                                             '\n\n': np.float32(-2.7019882),
                                             '!': np.float32(-6.508561),
                                             '!\n\n': np.float32(-8.167583),
                                             'さん': np.float32(-0.08028601),
                                             'さんの': np.float32(-7.432049),
                                             'サン': np.float32(-7.4604864),
                                             '！': np.float32(-7.488392),
                                             '！\n\n': np.float32(-7.848667)},
                                            {'': np.float32(-6.5023985),
                                             '\n': np.float32(-4.4653406),
                                             '\n\n': np.float32(-0.016384935),
                                             ' \n\n': np.float32(-8.3415785),
                                             '  \n': np.float32(-7.8905077),
                                             '  \n\n': np.float32(-8.05189),
                                             '。': np.float32(-7.018736),
                                             '。\n\n': np.float32(-7.8876543),
                                             'です': np.float32(-8.975427),
                                             '（': np.float32(-7.4214106)}]},
              'text': 'コックリさん'}],
 'created': 1741757929,
 'id': 'cmpl-48207e76-ced0-49b6-b4d2-038f0bebb0a4',
 'model': '/Users/tsutomunagayoshi/research/llm/local_o1/output/exp_test/checkpoint-1120_Q4_K_M.gguf',
 'object': 'text_completion',
 'usage': {'completion_tokens': 5, 'prompt_tokens': 1161, 'total_tokens': 1166}}
'''


'''
{'choices': [{'finish_reason': 'stop',
              'index': 0,
              'logprobs': None,
              'text': 'コックリさん'}],
 'created': 1741758091,
 'id': 'cmpl-085cc466-a042-4051-9216-857d0c128d8d',
 'model': '/Users/tsutomunagayoshi/research/llm/local_o1/output/exp_test/checkpoint-1120_Q4_K_M.gguf',
 'object': 'text_completion',
 'usage': {'completion_tokens': 5, 'prompt_tokens': 1161, 'total_tokens': 1166}}
'''