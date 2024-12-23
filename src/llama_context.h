#ifndef LLAMA_CONTEXT_H
#define LLAMA_CONTEXT_H

#include "llama.h"
#include "common.h"

#include "llama_model.h"
#include "sampling.h"
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include <godot_cpp/classes/worker_thread_pool.hpp>
#include <godot_cpp/templates/vector.hpp>
#include <algorithm>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/worker_thread_pool.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/templates/vector.hpp>
namespace godot {

struct completion_request {
	int id;
	godot::String prompt;
};

class LlamaContext : public Node {
	GDCLASS(LlamaContext, Node)

private:
	Ref<LlamaModel> model;
	llama_context  * g_ctx = nullptr;
    common_sampler    * g_smpl = nullptr;
	common_params     * g_params;
    int32_t n_len = 1024;
	int request_id = 0;
	Vector<completion_request> completion_requests;

	Ref<Thread> thread;
	Ref<Semaphore> semaphore;
	Ref<Mutex> mutex;
  std::vector<llama_token> context_tokens;
  bool exit_thread = false;
  bool init_success = false;

protected:
	static void _bind_methods();

public:
	void set_model(const Ref<LlamaModel> model);
	Ref<LlamaModel> get_model();
	void load_context();
	void start_thread();
	

	int request_completion(const String &prompt);
	void __thread_loop();

	uint32_t get_seed();
	void set_seed(uint32_t seed);
	uint32_t get_n_ctx();
	void set_n_ctx(uint32_t n_ctx);
  int32_t get_n_len();
  void set_n_len(int32_t n_len);
  float get_temperature();
  void set_temperature(float temperature);
  float get_top_p();
  void set_top_p(float top_p);
  float get_frequency_penalty();
  void set_frequency_penalty(float frequency_penalty);
  float get_presence_penalty();
  void set_presence_penalty(float presence_penalty);

	virtual PackedStringArray _get_configuration_warnings() const override;
	virtual void _enter_tree() override;
  virtual void _exit_tree() override;
	LlamaContext();
};
} //namespace godot

#endif