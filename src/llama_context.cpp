#include "llama_context.h"
#include "llama.h"
#include "llama_model.h"
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/worker_thread_pool.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

void LlamaContext::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_model", "model"), &LlamaContext::set_model);
	ClassDB::bind_method(D_METHOD("get_model"), &LlamaContext::get_model);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::OBJECT, "model", PROPERTY_HINT_RESOURCE_TYPE, "LlamaModel"), "set_model", "get_model");

	ClassDB::bind_method(D_METHOD("get_seed"), &LlamaContext::get_seed);
	ClassDB::bind_method(D_METHOD("set_seed", "seed"), &LlamaContext::set_seed);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");

	ClassDB::bind_method(D_METHOD("get_temperature"), &LlamaContext::get_temperature);
	ClassDB::bind_method(D_METHOD("set_temperature", "temperature"), &LlamaContext::set_temperature);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::FLOAT, "temperature"), "set_temperature", "get_temperature");

	ClassDB::bind_method(D_METHOD("get_top_p"), &LlamaContext::get_top_p);
	ClassDB::bind_method(D_METHOD("set_top_p", "top_p"), &LlamaContext::set_top_p);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::FLOAT, "top_p"), "set_top_p", "get_top_p");

	ClassDB::bind_method(D_METHOD("get_frequency_penalty"), &LlamaContext::get_frequency_penalty);
	ClassDB::bind_method(D_METHOD("set_frequency_penalty", "frequency_penalty"), &LlamaContext::set_frequency_penalty);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::FLOAT, "frequency_penalty"), "set_frequency_penalty", "get_frequency_penalty");

	ClassDB::bind_method(D_METHOD("get_presence_penalty"), &LlamaContext::get_presence_penalty);
	ClassDB::bind_method(D_METHOD("set_presence_penalty", "presence_penalty"), &LlamaContext::set_presence_penalty);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::FLOAT, "presence_penalty"), "set_presence_penalty", "get_presence_penalty");

	ClassDB::bind_method(D_METHOD("get_n_ctx"), &LlamaContext::get_n_ctx);
	ClassDB::bind_method(D_METHOD("set_n_ctx", "n_ctx"), &LlamaContext::set_n_ctx);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::INT, "n_ctx"), "set_n_ctx", "get_n_ctx");

	ClassDB::bind_method(D_METHOD("get_n_len"), &LlamaContext::get_n_len);
	ClassDB::bind_method(D_METHOD("set_n_len", "n_len"), &LlamaContext::set_n_len);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::INT, "n_len"), "set_n_len", "get_n_len");

	ClassDB::bind_method(D_METHOD("request_completion", "prompt"), &LlamaContext::request_completion);
	ClassDB::bind_method(D_METHOD("__thread_loop"), &LlamaContext::__thread_loop);

	// load_context
	ClassDB::bind_method(D_METHOD("load_context"), &LlamaContext::load_context);
	// start_thread
	ClassDB::bind_method(D_METHOD("start_thread"), &LlamaContext::start_thread);

	ADD_SIGNAL(MethodInfo("completion_generated", PropertyInfo(Variant::DICTIONARY, "chunk")));
}

LlamaContext::LlamaContext() {
	// ctx_params = llama_context_default_params();


    // llama_context * ctx = nullptr;
    // common_sampler * smpl = nullptr;


	int32_t n_threads = OS::get_singleton()->get_processor_count();
	
	common_params params;
	params.cpuparams.n_threads = n_threads;
    g_params = &params;
	common_init();
}

void LlamaContext::_enter_tree() {
	// TODO: remove this and use runtime classes once godot 4.3 lands, see https://github.com/godotengine/godot/pull/82554
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}



	// this->mutex.instantiate();
	// this->semaphore.instantiate();
	// this->thread.instantiate();

	llama_backend_init();
	llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED);
	
}

// need ctx, params, and model at this point
void LlamaContext::load_context() {


	UtilityFunctions::print(vformat("%s: Initializing llama context", __func__));
	if (model->model == NULL) {
		UtilityFunctions::printerr(vformat("%s: Failed to initialize llama context, model property not defined", __func__));
		return;
	}
	UtilityFunctions::print(vformat("%s: Model loaded", __func__));
	
	int32_t n_threads = OS::get_singleton()->get_processor_count();
	
	common_params params;
	params.cpuparams.n_threads = n_threads;
    
	g_params = &params;
	common_init();
	// IMPORTANT: NEED TO MAKE SURE THE G PARAMS ARE SET BEFORE THIS POINT
	if (&g_params == NULL) {
		UtilityFunctions::printerr(vformat("%s: Failed to initialize llama context, g_params not defined", __func__));
		return;
	}
	
	// TODO: figure out how to set the params from common_params
	// llama_context_params ctx_params = llama_context_params_from_common_params(*g_params);
	llama_context_params ctx_params = llama_context_default_params();
	UtilityFunctions::print(vformat("%s: Context created", __func__));

	
	g_ctx = llama_new_context_with_model(model->model, ctx_params); // we use this because the model will already be loaded by godot in memory
	// g_ctx = &ctx;
	if (g_ctx == NULL) {
		UtilityFunctions::printerr(vformat("%s: Failed to initialize llama context, null ctx", __func__));
		return;
	}
	
	
	UtilityFunctions::print(vformat("%s: Initializing llama sampler", __func__));
	// TODO: for some reason it thinks this is a grammar
	UtilityFunctions::print(vformat("%s: Setting grammar", __func__));
	UtilityFunctions::print(vformat("%s: this is the current grammar: %s", __func__, g_params->sparams.grammar.c_str()));

	UtilityFunctions::print(vformat("%s: Initialized 0", __func__));
	// print grammer
	UtilityFunctions::print(vformat("%s: %s", __func__, g_params->sparams.grammar.c_str()));
	g_smpl = common_sampler_init(model->model, g_params->sparams);


	UtilityFunctions::print(vformat("%s: Initialized 1", __func__));
	if (!g_smpl) {
		UtilityFunctions::printerr(vformat("%s: Failed to initialize llama context, null smpl", __func__));
		return;
	}

	UtilityFunctions::print(vformat("%s: Context initialized", __func__));
	init_success = true;
}

// everything should be initialized at this point
void LlamaContext::start_thread() {
	UtilityFunctions::print(vformat("%s: Starting thread", __func__));
	if (!init_success) {
		return;
	}
	this->thread.instantiate();	
	semaphore.instantiate();
	mutex.instantiate();
	UtilityFunctions::print(vformat("%s: Thread starting..", __func__));
	this->thread->start(callable_mp(this, &LlamaContext::__thread_loop));
	UtilityFunctions::print(vformat("%s: Thread started", __func__));
}

void LlamaContext::__thread_loop() {

	
	if (!init_success) {
		return;
	}
	while (true) {
		semaphore->wait();

		mutex->lock();
		if (exit_thread) {
			mutex->unlock();
			break;
		}
		if (completion_requests.size() == 0) {
			mutex->unlock();
			continue;
		}
		completion_request req = completion_requests.get(0);
		completion_requests.remove_at(0);
		mutex->unlock();

		UtilityFunctions::print(vformat("%s: Running completion for prompt id: %d", __func__, req.id));
		// get ctx from g_ctx
		
		UtilityFunctions::print(vformat("%s: Context loaded", __func__));
		UtilityFunctions::print(vformat("%s: Sampler loaded", __func__));
		std::vector<llama_token> request_tokens;	
		UtilityFunctions::print(vformat("%s: Request tokens: %s", __func__, req.prompt.utf8().get_data()));
		const std::string prompt_str = std::string(req.prompt.utf8().get_data());
		request_tokens = common_tokenize(model->model, prompt_str, true, true);
		UtilityFunctions::print(vformat("%s: Request tokens: %d", __func__, (int32_t)request_tokens.size()));

		size_t shared_prefix_idx = 0;
		auto diff = std::mismatch(context_tokens.begin(), context_tokens.end(), request_tokens.begin(), request_tokens.end());
		if (diff.first != context_tokens.end()) {
			shared_prefix_idx = std::distance(context_tokens.begin(), diff.first);
		} else {
			shared_prefix_idx = std::min(context_tokens.size(), request_tokens.size());
		}
		UtilityFunctions::print(vformat("%s: Shared prefix idx3: %d", __func__, (int32_t)shared_prefix_idx));
		// make sure that ctx is actually pointing to a memort address of type llama_context
		if (g_ctx == NULL) {
			UtilityFunctions::printerr(vformat("%s: Failed to request completion, ctx not initialized", __func__));
			continue;
		}
		// if the object ctx is pointing to is not of type llama_context, then this will fail
		if (g_ctx != nullptr) {
			// Example assuming `llama_context` has a member `int id;`
			UtilityFunctions::print(vformat("%s: Context id", __func__));
		}

		bool rm_success = llama_kv_cache_seq_rm(g_ctx, -1, shared_prefix_idx, -1);
		UtilityFunctions::print(vformat("%s: Removing tokens from kv cache", __func__));
		if (!rm_success) {
			UtilityFunctions::printerr(vformat("%s: Failed to remove tokens from kv cache", __func__));
			Dictionary response;
			response["id"] = req.id;
			response["error"] = "Failed to remove tokens from kv cache";
			call_thread_safe("emit_signal", "completion_generated", response);
			continue;
		}
		context_tokens.erase(context_tokens.begin() + shared_prefix_idx, context_tokens.end());
		request_tokens.erase(request_tokens.begin(), request_tokens.begin() + shared_prefix_idx);
		// print the two batch sizes
		UtilityFunctions::print(vformat("%s: Context tokens: %d", __func__, (int32_t)context_tokens.size()));
		UtilityFunctions::print(vformat("%s: Request tokens: %d", __func__, (int32_t)request_tokens.size()));
		UtilityFunctions::print(vformat("%s: N Batch: %d", __func__, g_params->n_batch));
		int32_t batch_size = std::min(g_params->n_batch, (int32_t)request_tokens.size());
		// min of 1
		batch_size = std::max(batch_size, 1);

		UtilityFunctions::print(vformat("%s: Batch size: %d", __func__, batch_size));
		llama_batch batch = llama_batch_init(batch_size, 0, 1);

		// chunk request_tokens into sequences of size batch_size
		UtilityFunctions::print(vformat("%s: Chunking request tokens", __func__));
		std::vector<std::vector<llama_token>> sequences;
		for (size_t i = 0; i < request_tokens.size(); i += batch_size) {
			sequences.push_back(std::vector<llama_token>(request_tokens.begin() + i, request_tokens.begin() + std::min(i + batch_size, request_tokens.size())));
		}

		UtilityFunctions::print(vformat("%s: Sequences: %d", __func__, (int32_t)sequences.size()));
		// for (auto sequence : sequences) {
		// 	for (auto token : sequence) {
		// 		printf("%s", common_token_to_piece(g_ctx, token, true).c_str());
		// 	}
		// }
		

		int curr_token_pos = context_tokens.size();
		bool decode_failed = false;
		UtilityFunctions::print(vformat("%s: Decoding sequences", __func__));

		// TODO: use the previous prompt so we dont need to re encode the same tokens
		for (size_t i = 0; i < sequences.size(); i++) {
			common_batch_clear(batch);

			std::vector<llama_token> sequence = sequences[i];

			for (size_t j = 0; j < sequence.size(); j++) {
				common_batch_add(batch, sequence[j], j + curr_token_pos, { 0 }, false);
			}

			curr_token_pos += sequence.size();

			if (i == sequences.size() - 1) {
				batch.logits[batch.n_tokens - 1] = true;
			}

			if (llama_decode(g_ctx, batch) != 0) {
				decode_failed = true;
				break;
			}
		}


		UtilityFunctions::print(vformat("%s: Request tokens: %d", __func__, (int32_t)request_tokens.size()));
		UtilityFunctions::print(vformat("%s: Batch tokens: %d", __func__, batch.n_tokens));
		UtilityFunctions::print(vformat("%s: Current token pos: %d", __func__, curr_token_pos));

		if (decode_failed) {
			Dictionary response;
			response["id"] = req.id;
			response["error"] = "llama_decode() failed";
			call_thread_safe("emit_signal", "completion_generated", response);
			continue;
		}

		context_tokens.insert(context_tokens.end(), request_tokens.begin(), request_tokens.end());
		UtilityFunctions::print(vformat("%s: Generating completions", __func__));
		while (true) {
			if (exit_thread) {
				return;
			}
			// UtilityFunctions::print(vformat("%s: Generating completions...", __func__));
			
			llama_token new_token_id =  common_sampler_sample(g_smpl, g_ctx, -1);
			// g_params->smpl->grmr is how to get grammar
            common_sampler_accept(g_smpl, new_token_id, /* accept_grammar= */ false);

			Dictionary response;
			response["id"] = req.id;

			context_tokens.push_back(new_token_id);

			bool eog = llama_token_is_eog(model->model, new_token_id);
			bool curr_eq_n_len = curr_token_pos == n_len;

			if (eog || curr_eq_n_len) {
				response["done"] = true;
				call_thread_safe("emit_signal", "completion_generated", response);
				break;
			}

	
			response["text"] = common_token_to_piece(g_ctx, new_token_id, true).c_str();
			response["done"] = false;
			call_thread_safe("emit_signal", "completion_generated", response);

			common_batch_clear(batch);

			common_batch_add(batch, new_token_id, curr_token_pos, { 0 }, true);

			curr_token_pos++;

			if (llama_decode(g_ctx, batch) != 0) {
				decode_failed = true;
				break;
			}
		}

		common_sampler_reset(g_smpl);

		if (decode_failed) {
			Dictionary response;
			response["id"] = req.id;
			response["error"] = "llama_decode() failed";
			call_thread_safe("emit_signal", "completion_generated", response);
			continue;
		}
	}
}

PackedStringArray LlamaContext::_get_configuration_warnings() const {
	PackedStringArray warnings;
	if (model == NULL) {
		warnings.push_back("Model resource property not defined");
	}
	return warnings;
}

int LlamaContext::request_completion(const String &prompt) {
	if (!init_success) {
		UtilityFunctions::printerr(vformat("%s: Failed to request completion, context not initialized", __func__));
		return -1;
	}
	int id = request_id++;

	UtilityFunctions::print(vformat("%s: Requesting completion for prompt id: %d", __func__, id));

	mutex->lock();
	completion_request req = { id, prompt };
	completion_requests.append(req);
	mutex->unlock();

	semaphore->post();

	return id;
}

void LlamaContext::set_model(const Ref<LlamaModel> p_model) {
	model = p_model;
}
Ref<LlamaModel> LlamaContext::get_model() {
	return model;
}

uint32_t LlamaContext::get_seed() {
	return g_params->sparams.seed;
}
void LlamaContext::set_seed(uint32_t seed) {
	g_params->sparams.seed = seed;
}

uint32_t LlamaContext::get_n_ctx() {
	return g_params->n_ctx;
}
void LlamaContext::set_n_ctx(uint32_t n_ctx) {
	g_params->n_ctx = n_ctx;
}

int32_t LlamaContext::get_n_len() {
	return n_len;
}
void LlamaContext::set_n_len(int32_t n_len) {
	this->n_len = n_len;
}

float LlamaContext::get_temperature() {
	return g_params->sparams.temp;
}
void LlamaContext::set_temperature(float temperature) {
	g_params->sparams.temp = temperature;
}

float LlamaContext::get_top_p() {
	return g_params->sparams.top_p;
}
void LlamaContext::set_top_p(float top_p) {
	g_params->sparams.top_p = top_p;
}

float LlamaContext::get_frequency_penalty() {
	return g_params->sparams.penalty_freq;
}
void LlamaContext::set_frequency_penalty(float frequency_penalty) {
	g_params->sparams.penalty_freq = frequency_penalty;
}

float LlamaContext::get_presence_penalty() {
	return g_params->sparams.penalty_present;
}
void LlamaContext::set_presence_penalty(float presence_penalty) {
	g_params->sparams.penalty_present = presence_penalty;
}

void LlamaContext::_exit_tree() {
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	mutex->lock();
	exit_thread = true;
	mutex->unlock();

	semaphore->post();

	thread->wait_to_finish();

	if (g_ctx) {
		llama_free(g_ctx);
	}
	if (g_smpl) {
		common_sampler_free(g_smpl);
	}
	llama_backend_free();
}