#ifndef __THREAD_SYNCHRONIZER_H__
#define __THREAD_SYNCHRONIZER_H__
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace std;

/**
 * Wait & notification mechanism for a single waiter & a single notifier.
 */
class WaitNotifySingle {

    mutex              m_mutex;
    condition_variable m_cond_var;
    atomic_bool        m_cond_var_flag;
    atomic_bool        m_waiting;

    atomic_bool        m_terminating;

  public:
    WaitNotifySingle()
        :m_cond_var_flag (false)
        ,m_waiting       (false)
        ,m_terminating   (false)
        {;}

    ~WaitNotifySingle(){
        terminate();
    }

    /** 
     * @brief lets the waiter know that it should terminate the thread execution.
     */
    void terminate() {

        unique_lock<mutex> lock( m_mutex, defer_lock );
        lock.lock();
        m_terminating.store( true, memory_order_release );
        lock.unlock();
        m_cond_var.notify_one();
    }

    /**
     * @brief the waiter can check if it should terminate its execution.
     */
    bool isTerminating() {
        return m_terminating.load( memory_order_acquire );
    }

    /** 
     * @brief give the waiting thread a go ahead.
     *        It also waits in a spin lock if the waiter is not yet in wait().
     */
    inline void notify() {
        if ( !m_terminating.load( memory_order_acquire ) ) {

            unique_lock<mutex> lock( m_mutex, defer_lock );

            while(    !m_waiting.    load( memory_order_acquire )
                   && !m_terminating.load( memory_order_acquire ) ) {;}
            lock.lock();       
            m_waiting.      store( false, memory_order_release );
            m_cond_var_flag.store( true,  memory_order_release );
            lock.unlock();
            m_cond_var.notify_one();
        }
    }

    /**
     * @brief waits until the notifier calls notify().
     */
    inline void wait() {
        if ( !m_terminating.load( memory_order_acquire ) ) {
            unique_lock<mutex> lock( m_mutex, defer_lock );
            lock.lock();
            m_waiting.store( true, memory_order_release );
            m_cond_var.wait( lock, [&] { return    m_cond_var_flag.load( memory_order_acquire )
                                                || m_terminating.  load( memory_order_acquire ); } );
            m_cond_var_flag.store( false, memory_order_release );
            lock.unlock();
        }
    }
};


/**
 * Wait & notification mechanism for multiple waiters & a single notifier.
 * It is mainly used with WaitNotifyMultipleNotifiers to form a parallel
 * execution of a thread group.
 */
class WaitNotifyMultipleWaiters {

    mutex                m_mutex;
    condition_variable   m_cond_var;
    vector<atomic_bool*> m_cond_var_flags;
    atomic_int           m_num_waiting;

    atomic_bool          m_terminating;

    const int            m_num_waiters;

  public:

    /**
     * @param num_waiters (in): number of waiters must be fixed at the construction
     */
    WaitNotifyMultipleWaiters( const int num_waiters )
        :m_num_waiting   (0)
        ,m_terminating   (false)
        ,m_num_waiters   (num_waiters)
    {
        for ( int i = 0; i < m_num_waiters; i++ ) {

            m_cond_var_flags.emplace_back( new atomic_bool() );
        }

        for ( int i = 0; i < m_num_waiters; i++ ) {

            m_cond_var_flags[i]->store( false,  memory_order_release );
        }
    }

    ~WaitNotifyMultipleWaiters(){
        terminate();
        for ( int i = 0; i < m_num_waiters; i++ ) {
            delete m_cond_var_flags[i];
        }
    }

    /** 
     * @brief lets the waiters know that they should terminate the thread execution.
     */
    void terminate() {

        unique_lock<mutex> lock( m_mutex, defer_lock );
        lock.lock();
        m_terminating.store( true, memory_order_release );
        lock.unlock();
        m_cond_var.notify_all();
    }

    /**
     * @brief the waiters can check if it should terminate its execution.
     */
    bool isTerminating() {
        return m_terminating.load( memory_order_acquire );
    }


    /** 
     * @brief give the waiting threads a go ahead.
     *        It also waits in a spin lock if all the waiters are not yet in wait().
     */
    inline void notify() {
        if ( !m_terminating.load( memory_order_acquire ) ) {

            unique_lock<mutex> lock( m_mutex, defer_lock );

            while(     m_num_waiting.load( memory_order_acquire) < m_num_waiters
                   && !m_terminating.load( memory_order_acquire ) ) {;}
            lock.lock();
            m_num_waiting.store( 0, memory_order_release );
            for ( auto* f : m_cond_var_flags ) {
                f->store( true,  memory_order_release );
            }
            lock.unlock();
            m_cond_var.notify_all();
        }
    }


    /**
     * @brief waits until the notifier calls notify().
     * 
     * @param thread_id (in): the number that uniquely identifies the thread. 0 <= thread_id < m_num_waiters.
     */
    inline void wait( const int thread_id ) {
        if ( !m_terminating.load( memory_order_acquire ) ) {

            unique_lock<mutex> lock( m_mutex, defer_lock );
            lock.lock();

            m_num_waiting.fetch_add( 1, memory_order_acq_rel );

            m_cond_var.wait( lock, [&] { return    m_cond_var_flags[ thread_id ]->load( memory_order_acquire )
                                                || m_terminating.load( memory_order_acquire ); } );

            m_cond_var_flags[ thread_id ]->store( false, memory_order_release );

            lock.unlock();

        }
    }
};


/**
 * Wait & notification mechanism for a single waiter & multiple notifiers.
 * It is mainly used with WaitNotifyMultipleNotifiers to form a parallel
 * execution of a thread group.
 */
class WaitNotifyMultipleNotifiers {


    mutex              m_mutex;
    condition_variable m_cond_var;
    atomic_int         m_num_notified;
    atomic_bool        m_waiting;

    atomic_bool        m_terminating;

    const int          m_num_notifiers;

  public:

    /**
     * @param num_notifiers (in): number of notifiers must be fixed at the construction
     */
    WaitNotifyMultipleNotifiers( const int num_notifiers )
        :m_num_notified  (0)
        ,m_waiting       (false)
        ,m_terminating   (false)
        ,m_num_notifiers (num_notifiers)
    {;}


    ~WaitNotifyMultipleNotifiers(){
        terminate();
    }

    /** 
     * @brief lets the waiter know that they should terminate the thread execution.
     */
    void terminate() {

        unique_lock<mutex> lock( m_mutex, defer_lock );
        lock.lock();
        m_terminating.store( true, memory_order_release );
        lock.unlock();
        m_cond_var.notify_one();
    }

    /**
     * @brief the waiter can check if it should terminate its execution.
     */
    bool isTerminating() {
        return m_terminating.load( memory_order_acquire );
    }

    /** 
     * @brief give the waiting threads a go ahead.
     *        It also waits in a spin lock if the waiter is not yet in wait().
     */
    inline void notify() {
        if ( !m_terminating.load( memory_order_acquire ) ) {

            unique_lock<mutex> lock( m_mutex, defer_lock );

            while(    !m_waiting.    load( memory_order_acquire )
                   && !m_terminating.load( memory_order_acquire ) ) {;}

            lock.lock();       

            m_num_notified.fetch_add( 1,  memory_order_acq_rel );
            lock.unlock();

            m_cond_var.notify_one();

        }
    }

    /**
     * @brief waits until all the notifier call notify().
     */
    inline void wait() {

        if ( !m_terminating.load( memory_order_acquire ) ) {

            unique_lock<mutex> lock( m_mutex, defer_lock );

            lock.lock();
            m_waiting.store( true, memory_order_release );

            m_cond_var.wait( lock, [&] { return    m_num_notified.load( memory_order_acquire ) == m_num_notifiers
                                                || m_terminating.load( memory_order_acquire ) ; } );

            m_waiting.     store( false, memory_order_release );
            m_num_notified.store( 0, memory_order_release );

            lock.unlock();

        }
    }
};


/**
 * Wait & notification mechanism for N waiters & N notifiers.
 */
class WaitNotifyNxN {

    mutex                m_mutex;
    condition_variable   m_cond_var;
    vector<atomic_bool*> m_cond_var_flags;
    atomic_int           m_num_waiting;
    atomic_int           m_num_notifying;

    atomic_bool          m_terminating;

    const int            m_num_participants;

  public:

    /**
     * @param num_participants (in): number of notifiers/waiters  must be fixed at the construction
     */
    WaitNotifyNxN( const int num_participants )
        :m_num_waiting      (0)
        ,m_num_notifying    (0)
        ,m_terminating      (false)
        ,m_num_participants (num_participants)
    {
        for ( int i = 0; i < m_num_participants; i++ ) {

            m_cond_var_flags.emplace_back( new atomic_bool() );
        }

        for ( int i = 0; i < m_num_participants; i++ ) {

            m_cond_var_flags[i]->store( false,  memory_order_release );
        }
    }

    ~WaitNotifyNxN(){
        terminate();
        for ( int i = 0; i < m_num_participants; i++ ) {
            delete m_cond_var_flags[i];
        }
    }


    /** 
     * @brief give the waiting threads a go ahead.
     *        It also waits in a spin lock if not al the waiters are yet in wait().
     */
    inline void notify() {
        if ( !m_terminating.load( memory_order_acquire ) ) {

            unique_lock<mutex> lock( m_mutex, defer_lock );

            while(     m_num_waiting.load( memory_order_acquire ) < m_num_participants
                   && !m_terminating.load( memory_order_acquire ) ){;}
            lock.lock();       
            auto v = m_num_notifying.fetch_add( 1, memory_order_acq_rel );
            if ( v + 1 ==  m_num_participants ) {
                for ( auto* f : m_cond_var_flags ) {
                    f->store( true,  memory_order_release );
                }
                m_num_notifying.store( 0, memory_order_release );
                m_num_waiting.  store( 0, memory_order_release );
                lock.unlock();
                m_cond_var.notify_all();
            }
            else {
                lock.unlock();
            }
        }
    }

    /**
     * @brief waits until all the notifier call notify().
     * 
     * @param thread_id (in): the number that uniquely identifies the thread. 0 <= thread_id < m_num_participants.
     */
    inline void wait( const int thread_id ) {
        if ( !m_terminating.load( memory_order_acquire ) ) {

            unique_lock<mutex> lock( m_mutex, defer_lock );
            lock.lock();

            m_num_waiting.fetch_add( 1, memory_order_acq_rel );
            m_cond_var.wait( lock, [&] { return    m_terminating.load( memory_order_acquire )
                                                || m_cond_var_flags[ thread_id ]->load( memory_order_acquire ); } );
            m_cond_var_flags[ thread_id ]->store( false, memory_order_release );
            lock.unlock();
        }
    }

    /** 
     * @brief lets the waiters know that they should terminate the thread execution.
     */
    void terminate() {

        unique_lock<mutex> lock( m_mutex, defer_lock );
        lock.lock();
        m_terminating.store( true, memory_order_release );
        lock.unlock();
        m_cond_var.notify_all();
    }


    /**
     * @brief the waiters can check if it should terminate its execution.
     */
    bool isTerminating() {
        return m_terminating.load( memory_order_acquire );
    }
};


/**
 * Synchronization mechanism among multipel threads running in parallel in a group.
 * it works as __syncthreads() in CUDA in a block.
 */
class WaitNotifyEachOther {

    mutex                m_mutex;
    condition_variable   m_cond_var;
    vector<atomic_bool*> m_cond_var_flags;
    atomic_bool          m_is_ready;
    atomic_int           m_num_waiting;

    atomic_bool          m_terminating;

    const int            m_num_participants;

  public:

    /**
     * @param num_participants (in): number of threads in the group must be fixed at construction.
     */
    WaitNotifyEachOther( const int num_participants )
        :m_is_ready         (false)
        ,m_num_waiting      (0)
        ,m_terminating      (false)
        ,m_num_participants (num_participants)
    {
        for ( int i = 0; i < m_num_participants; i++ ) {

            m_cond_var_flags.emplace_back( new atomic_bool() );
        }

        for ( int i = 0; i < m_num_participants; i++ ) {

            m_cond_var_flags[i]->store( false,  memory_order_release );
        }
        m_is_ready.store( true,  memory_order_release );
    }

    ~WaitNotifyEachOther(){
        terminate();
        for ( int i = 0; i < m_num_participants; i++ ) {
            delete m_cond_var_flags[i];
        }
    }


    /**
     * @brief waits until all the other participating threads calls syncThreads().
     * 
     * @param thread_id (in): the number that uniquely identifies the thread. 0 <= thread_id < m_num_participants.
     */
    inline void syncThreads( const int thread_id ) {

        if ( !m_terminating.load( memory_order_acquire ) ) {

            unique_lock<mutex> lock( m_mutex, defer_lock );

            while(    !m_is_ready.load( memory_order_acquire ) 
                   && !m_terminating.load( memory_order_acquire ) ){;}
            lock.lock();
            auto prev_val = m_num_waiting.fetch_add( 1, memory_order_acq_rel );
            if ( prev_val == m_num_participants - 1 ) {

                // last thread to syncThreads.
                m_is_ready.store( false, memory_order_release );
                m_num_waiting.fetch_add( -1, memory_order_acq_rel );
                for ( int i = 0 ; i < m_cond_var_flags.size() ; i++ ) {
                    if ( i != thread_id ) {
                        m_cond_var_flags[i]->store( true,  memory_order_release );
                    }
                }
                lock.unlock();
                m_cond_var.notify_all();
            }
            else {
                m_cond_var.wait( lock, [&] { return    m_terminating.load( memory_order_acquire )
                                                    || m_cond_var_flags[ thread_id ]->load( memory_order_acquire ); } );

                m_cond_var_flags[ thread_id ]->store( false, memory_order_release );

                auto prev_val_after_wait = m_num_waiting.fetch_add( -1, memory_order_acq_rel );

                if (prev_val_after_wait == 1) {

                    m_is_ready.store( true, memory_order_release );
                }
                lock.unlock();
            }
        }
    }

    /** 
     * @brief lets all the participaint threads know that they should terminate the thread execution.
     */
    void terminate() {

        unique_lock<mutex> lock( m_mutex, defer_lock );
        lock.lock();
        m_terminating.store( true, memory_order_release );
        lock.unlock();
        m_cond_var.notify_all();
    }

    /**
     * @brief the participating threads can check if it should terminate its execution.
     */
    bool isTerminating() {
        return m_terminating.load( memory_order_acquire );
    }
};


#endif /*__THREAD_SYNCHRONIZER_H__*/
