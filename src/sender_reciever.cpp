/*
 *  Copyright 2022 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// This is a toy implementation of the core parts of the C++ std::execution
// proposal (aka, Executors, http://wg21.link/P2300). It is intended to be a
// learning tool only. THIS CODE IS NOT SUITABLE FOR ANY USE.

#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

// Some utility code
///////////////////////////////////////////
std::string get_thread_id() {
    std::stringstream sout;
    sout.imbue(std::locale::classic());
    sout << "0x" << std::hex << std::this_thread::get_id();
    return sout.str();
}

struct immovable {
    immovable() = default;
    immovable(immovable&&) = delete;
};

struct none {};

// In this toy implementation, a sender can only complete with a single value.
template <class Snd>
using sender_result_t = typename Snd::result_t;

///////////////////////////////////////////
// just(T) sender factory
///////////////////////////////////////////
template <class T, class Rcvr>
struct just_operation : immovable {
    T value_;
    Rcvr rcvr_;

    friend void start(just_operation& self) {
        set_value(self.rcvr_, self.value_);
    }
};

template <class T>
struct just_sender {
    using result_t = T;

    T value_;

    template <class Rcvr>
    friend just_operation<T, Rcvr> connect(just_sender self, Rcvr rcvr) {
        return {{}, self.value_, rcvr};
    }
};

///////////////////////////////////////////
// then(Sender, Function) sender adaptor
///////////////////////////////////////////
template <class Fun, class Rcvr>
struct then_receiver {
    Fun fun_;
    Rcvr rcvr_;

    friend void set_value(then_receiver self, auto val) try {
        set_value(self.rcvr_, self.fun_(val));
    } catch(...) {
        set_error(self.rcvr_, std::current_exception());
    }
    friend void set_error(then_receiver self, std::exception_ptr err) {
        set_error(self.rcvr_, err);
    }
    friend void set_stopped(then_receiver self) {
        set_stopped(self.rcvr_);
    }
};

template <class PrevSnd, class Fun>
struct then_sender {
    using prev_result_t = sender_result_t<PrevSnd>;
    using result_t = std::invoke_result_t<Fun, prev_result_t>;

    PrevSnd prev_;
    Fun fun_;

    template <class Rcvr>
    friend auto connect(then_sender self, Rcvr rcvr) {
        return connect(self.prev_, then_receiver<Fun, Rcvr>{self.fun_, rcvr});
    }
};

template <class PrevSnd, class Fun>
then_sender<PrevSnd, Fun> then(PrevSnd prev, Fun fun) {
    return {prev, fun};
}

///////////////////////////////////////////
// sync_wait() sender consumer
///////////////////////////////////////////
struct sync_wait_data {
    std::mutex mtx;
    std::condition_variable cv;
    std::exception_ptr err;
    bool completed = false;
};

template <class T>
struct sync_wait_receiver {
    sync_wait_data& data_;
    std::optional<T>& value_;

    friend void set_value(sync_wait_receiver self, T val) {
        std::unique_lock lk(self.data_.mtx);
        self.value_.emplace(val);
        self.data_.completed = true;
        self.data_.cv.notify_all();
    }
    friend void set_error(sync_wait_receiver self, std::exception_ptr err) {
        std::unique_lock lk(self.data_.mtx);
        self.data_.err = err;
        self.data_.completed = true;
        self.data_.cv.notify_all();
    }
    friend void set_stopped(sync_wait_receiver self) {
        std::unique_lock lk(self.data_.mtx);
        self.data_.completed = true;
        self.data_.cv.notify_all();
    }
};

template <class Snd>
std::optional<sender_result_t<Snd>> sync_wait(Snd snd) {
    using T = sender_result_t<Snd>;

    sync_wait_data data;
    std::optional<T> value;

    auto op = connect(snd, sync_wait_receiver<T>{data, value});
    start(op);

    std::unique_lock lk{data.mtx};
    data.cv.wait(lk, [&data]{ return data.completed; });

    if (data.err)
        std::rethrow_exception(data.err);

    return value;
}

///////////////////////////////////////////
// run_loop execution context
///////////////////////////////////////////
struct run_loop : private immovable {
private:
    struct operation_base  {
        operation_base* next_ = this;
        virtual void execute() {}
    };

    template <class Rcvr>
    struct operation : operation_base {
        Rcvr rcvr_;
        run_loop& ctx_;

        operation(Rcvr rcvr, run_loop& ctx)
          : rcvr_(rcvr), ctx_(ctx) {}

        void execute() override final {
            std::printf("Running task on thread: %s\n", get_thread_id().c_str());
            set_value(rcvr_, none{});
        }

        friend void start(operation& self) {
            self.start_();
        }

        void start_() {
            ctx_.push_back_(this);
        }
    };
    template<class Rcvr>
    struct AsyncOperation:operation_base{
       
        struct heap_op_state_reciever{
            AsyncOperation<Rcvr>* op_state{nullptr};
            heap_op_state_reciever(AsyncOperation<Rcvr>* op):op_state(op){}
            template<class T>
            void set_value( T val) {
                op_state->rcvr_.set_value(val);
                delete op_state;
            }
            void set_error(std::exception_ptr err) {
                set_error(op_state->rcvr_,err);
                delete op_state;
            }
            void set_stopped() {
                set_stopped(op_state->rcvr_);
                delete op_state;
            }
        };
        AsyncOperation(Rcvr rcvr, run_loop& ctx)
        : rcvr_(rcvr),ctx_(ctx),
          heap_rcvr(this){}
        void execute() override final {
            std::printf("Running task on thread: %s\n", get_thread_id().c_str());
            heap_rcvr.set_value(none{});
        }

        friend void start(AsyncOperation& self) {
            self.start_();
        }

        void start_() {
            ctx_.push_back_(this);
        }
        Rcvr rcvr_;
        run_loop& ctx_;
        heap_op_state_reciever heap_rcvr;
    };

    operation_base head_;
    operation_base* tail_ = &head_;
    bool finish_ = false;
    std::mutex mtx_;
    std::condition_variable cv_;

    void push_back_(operation_base* op) {
        std::unique_lock lk(mtx_);
        op->next_ = &head_;
        tail_->next_= op;
        tail_ =tail_->next_;
        cv_.notify_one();
    }

    operation_base* pop_front_() {
        std::unique_lock lk(mtx_);
        // loop while the queue is empty:
        for (; head_.next_ == &head_; cv_.wait(lk))
            if (finish_)
                return nullptr;
        auto ret= std::exchange(head_.next_, head_.next_->next_);
        if(head_.next_==&head_){
            tail_ = &head_;
        }
        return ret;
    }
    
   
    

    struct sender {
        using result_t = none;
        run_loop& ctx_;

        template <class Rcvr>
        friend operation<Rcvr> connect(sender self, Rcvr rcvr) {
            return self.connect_(rcvr);
        }
        template <class Rcvr,class Allocator>
        friend AsyncOperation<Rcvr>& connect(sender self, Rcvr rcvr,Allocator alloc) {
            return self.connect_(rcvr,alloc);
        }
        template <class Rcvr,class Allocator>
        AsyncOperation<Rcvr>& connect_(Rcvr rcvr,Allocator ) {
            auto async_op = new AsyncOperation<Rcvr>(rcvr,ctx_);
            return *async_op;
        }
        template <class Rcvr>
        operation<Rcvr> connect_(Rcvr rcvr) {
            return {rcvr, ctx_};
        }
    };

    struct scheduler {
        run_loop& ctx_;

        friend sender schedule(scheduler self) {
            return {self.ctx_};
        }
    };

public:
    void run() {
        while (auto* op = pop_front_())
            op->execute();
    }

    scheduler get_scheduler() {
        return {*this};
    }

    void finish() {
        std::unique_lock lk(mtx_);
        finish_ = true;
        cv_.notify_all();
    }

};

///////////////////////////////////////////
// thread_context execution context
///////////////////////////////////////////
class thread_context : immovable {
    run_loop loop_;
    std::thread thread_;

public:
    thread_context()
      : thread_([this]{ loop_.run(); })
    {}

    void finish() {
        loop_.finish();
    }

    void join() {
        thread_.join();
    }

    auto get_scheduler() {
        return loop_.get_scheduler();
    }
};
template<class Func>
struct func_reciever{
        Func fun_;
        friend void set_value(func_reciever self, auto val) try {
            self.fun_(val);
        } catch(...) {
            std::printf("exited with error");
        }
        friend void set_error(func_reciever self, std::exception_ptr err) {
            std::printf("exited with error");
        }
        friend void set_stopped(func_reciever self) {
            
        }
        void set_value(auto val) try {
            fun_(val);
        } catch(...) {
            std::printf("exited with error");
        }

    };
template<class Func>
auto reciever_of(Func func){
    
    return func_reciever<Func>{func};
}
//
// start test code
//
struct Alloc{};
int main() {
    thread_context ctx;
    std::printf("main thread: %s\n", get_thread_id().c_str());

    //just_sender<int> first{42};
    auto sh= schedule(ctx.get_scheduler());
    auto first = then(sh, [](auto) { return 42;} );
    auto next = then(first, [](int i) {return i+1;} );
    auto last = then(next, [&](int i) {
        auto& next = connect(sh,reciever_of([](auto i){
            std::printf("will print this?\n");
            return 1;
        }),Alloc{});
        start(next);
        return i;
    } );

    int i = sync_wait(last).value();
    std::printf("result: %d\n", i);
    ctx.finish();
    ctx.join();
}
