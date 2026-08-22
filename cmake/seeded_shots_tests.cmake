# Seeded-SHOTS cross-host replay probe.
#
# Included from the root QSIM_BUILD_TESTS block.  The control-plane guard keeps
# this target honest on platforms where POSIX control transport is disabled.

if(QSIM_ENABLE_CONTROL_PLANE)
    add_executable(seeded_shots_replay_probe
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/seeded_shots_replay_probe.c)
    target_link_libraries(seeded_shots_replay_probe
        PRIVATE quantumsim ${MATH_LIBRARY} Threads::Threads)
endif()
