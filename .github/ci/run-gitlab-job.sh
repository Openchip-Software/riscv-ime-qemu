#!/bin/bash
#
# Run an upstream GitLab CI job inside its container.
#
# Each function below repeats the "script:" of the GitLab template with
# the same name in .gitlab-ci.d/. Keep them in sync with those templates
# when rebasing onto a new QEMU release.
#
# Usage: run-gitlab-job.sh TEMPLATE
#
# TEMPLATE is one of: native, cross-system, cross-user, cross-accel.
#
# The job variables come from the environment and have the names used in
# .gitlab-ci.d/: TARGETS, CONFIGURE_ARGS, EXTRA_CONFIGURE_OPTS and
# MAKE_CHECK_ARGS. JOB_MAKE replaces $MAKE when a job sets MAKE.
# TEST_MAKE_CHECK_ARGS and TEST_SCRIPT come from the test jobs that
# upstream runs on the build directory of a native build job.
# FUNCTIONAL_TESTS lists the functional tests, by meson test name, to run
# after them.
#
# SPDX-License-Identifier: GPL-2.0-or-later

set -ex

# .base_meson_job_template
JOBS=$(expr $(nproc) + 1)
MAKE=${JOB_MAKE:-$MAKE}

# The checkout belongs to the runner user, not to root in the container.
git config --global --add safe.directory "$PWD"

# .native_build_job_template
native_build_job() {
    mkdir build
    cd build
    ../configure --enable-werror --disable-docs --enable-fdt=system \
        --disable-debug-info \
        ${TARGETS:+--target-list="$TARGETS"} \
        $CONFIGURE_ARGS ||
        { cat config.log meson-logs/meson-log.txt && exit 1; }
    $MAKE -j"$JOBS"
    if test -n "$MAKE_CHECK_ARGS"; then
        $MAKE -j"$JOBS" $MAKE_CHECK_ARGS
    fi
}

# .native_test_job_template, and the "script:" of the crash-test-* jobs.
# Upstream runs these in a separate job on the build artifacts; here they
# run in the build directory that native_build_job left behind.
native_test_job() {
    if test -n "$TEST_MAKE_CHECK_ARGS"; then
        $MAKE NINJA=":" MTESTARGS="--no-suite doc" $TEST_MAKE_CHECK_ARGS
    fi
    if test -n "$TEST_SCRIPT"; then
        eval "$TEST_SCRIPT"
    fi
}

# .functional_test_job_template, for the tests in FUNCTIONAL_TESTS only.
# Upstream runs "make check-functional", which runs every functional test
# and first downloads the assets of all of them.
functional_test_job() {
    if test -n "$FUNCTIONAL_TESTS"; then
        export QEMU_TEST_ALLOW_UNTRUSTED_CODE=1
        export QEMU_TEST_CACHE_DIR="$(dirname "$PWD")/functional-cache"
        pyvenv/bin/meson test --setup thorough --no-rebuild \
            --print-errorlogs $FUNCTIONAL_TESTS
    fi
}

# .cross_system_build_job
cross_system_build_job() {
    mkdir build
    cd build
    ../configure --enable-werror --disable-docs --enable-fdt=system \
        --disable-user $QEMU_CONFIGURE_OPTS $EXTRA_CONFIGURE_OPTS \
        --target-list-exclude="arm-softmmu
          i386-softmmu microblaze-softmmu mips-softmmu mipsel-softmmu
          mips64-softmmu ppc-softmmu riscv32-softmmu sh4-softmmu
          sparc-softmmu xtensa-softmmu $CROSS_SKIP_TARGETS"
    make -j"$JOBS" all check-build
    if test -n "$MAKE_CHECK_ARGS"; then
        $MAKE -j"$JOBS" $MAKE_CHECK_ARGS
    fi
}

# .cross_user_build_job
cross_user_build_job() {
    mkdir build
    cd build
    ../configure --enable-werror --disable-docs $QEMU_CONFIGURE_OPTS \
        --disable-system --target-list-exclude="aarch64_be-linux-user
          alpha-linux-user m68k-linux-user microblazeel-linux-user
          or1k-linux-user ppc-linux-user sparc-linux-user
          xtensa-linux-user $CROSS_SKIP_TARGETS"
    make -j"$JOBS" all check-build
    if test -n "$MAKE_CHECK_ARGS"; then
        $MAKE -j"$JOBS" $MAKE_CHECK_ARGS
    fi
}

# .cross_accel_build_job
cross_accel_build_job() {
    mkdir build
    cd build
    ../configure --enable-werror --disable-docs $QEMU_CONFIGURE_OPTS \
        --disable-tools --enable-${ACCEL:-kvm} $EXTRA_CONFIGURE_OPTS
    make -j"$JOBS" all check-build
    if test -n "$MAKE_CHECK_ARGS"; then
        $MAKE -j"$JOBS" $MAKE_CHECK_ARGS
    fi
}

case "$1" in
native)
    native_build_job
    native_test_job
    functional_test_job
    ;;
cross-system)
    cross_system_build_job
    ;;
cross-user)
    cross_user_build_job
    ;;
cross-accel)
    cross_accel_build_job
    ;;
*)
    echo "usage: $0 native|cross-system|cross-user|cross-accel" >&2
    exit 1
    ;;
esac
