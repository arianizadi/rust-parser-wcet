	.text
	.file	"linear.32b9f2ca19ad7312-cgu.0"
	.hidden	_ZN3std2rt10lang_start17hb92c0478cf91f2feE # -- Begin function _ZN3std2rt10lang_start17hb92c0478cf91f2feE
	.globl	_ZN3std2rt10lang_start17hb92c0478cf91f2feE
	.p2align	4, 0x90
	.type	_ZN3std2rt10lang_start17hb92c0478cf91f2feE,@function
_ZN3std2rt10lang_start17hb92c0478cf91f2feE: # @_ZN3std2rt10lang_start17hb92c0478cf91f2feE
	.cfi_startproc
# %bb.0:                                # %start
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	%ecx, %eax
	movq	%rdx, %rcx
	movq	%rsi, %rdx
                                        # kill: def $al killed $al killed $eax
	movq	%rdi, (%rsp)
	movq	%rsp, %rdi
	movabsq	$.Lvtable.0, %rsi
	movzbl	%al, %r8d
	callq	*_ZN3std2rt19lang_start_internal17h15895544e2012228E@GOTPCREL(%rip)
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	_ZN3std2rt10lang_start17hb92c0478cf91f2feE, .Lfunc_end0-_ZN3std2rt10lang_start17hb92c0478cf91f2feE
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h433b39b893cf1495E
	.type	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h433b39b893cf1495E,@function
_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h433b39b893cf1495E: # @"_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h433b39b893cf1495E"
	.cfi_startproc
# %bb.0:                                # %start
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	(%rdi), %rdi
	callq	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17ha1bd341b6ffd4013E
	callq	_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17h4b8822d538ca828fE
	movzbl	%al, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h433b39b893cf1495E, .Lfunc_end1-_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h433b39b893cf1495E
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN3std3sys9backtrace28__rust_begin_short_backtrace17ha1bd341b6ffd4013E
	.type	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17ha1bd341b6ffd4013E,@function
_ZN3std3sys9backtrace28__rust_begin_short_backtrace17ha1bd341b6ffd4013E: # @_ZN3std3sys9backtrace28__rust_begin_short_backtrace17ha1bd341b6ffd4013E
	.cfi_startproc
# %bb.0:                                # %start
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	_ZN4core3ops8function6FnOnce9call_once17h6f82b81b85a088f7E
	#APP
	#NO_APP
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17ha1bd341b6ffd4013E, .Lfunc_end2-_ZN3std3sys9backtrace28__rust_begin_short_backtrace17ha1bd341b6ffd4013E
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN4core3fmt2rt8Argument11new_display17h550d7fff055c1ad2E
	.type	_ZN4core3fmt2rt8Argument11new_display17h550d7fff055c1ad2E,@function
_ZN4core3fmt2rt8Argument11new_display17h550d7fff055c1ad2E: # @_ZN4core3fmt2rt8Argument11new_display17h550d7fff055c1ad2E
	.cfi_startproc
# %bb.0:                                # %start
	movq	%rdi, %rax
	movq	%rsi, -16(%rsp)
	movq	_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$i32$GT$3fmt17h64ecadec32fa3e77E@GOTPCREL(%rip), %rcx
	movq	%rcx, -8(%rsp)
	movq	-16(%rsp), %rcx
	movq	%rcx, (%rdi)
	movq	-8(%rsp), %rcx
	movq	%rcx, 8(%rdi)
	retq
.Lfunc_end3:
	.size	_ZN4core3fmt2rt8Argument11new_display17h550d7fff055c1ad2E, .Lfunc_end3-_ZN4core3fmt2rt8Argument11new_display17h550d7fff055c1ad2E
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN4core3fmt9Arguments6new_v117heae077875e0c9aa4E
	.type	_ZN4core3fmt9Arguments6new_v117heae077875e0c9aa4E,@function
_ZN4core3fmt9Arguments6new_v117heae077875e0c9aa4E: # @_ZN4core3fmt9Arguments6new_v117heae077875e0c9aa4E
	.cfi_startproc
# %bb.0:                                # %start
	movq	%rdi, %rax
	movq	%rsi, (%rdi)
	movq	$2, 8(%rdi)
	movq	.L__unnamed_1, %rsi
	movq	.L__unnamed_1+8, %rcx
	movq	%rsi, 32(%rdi)
	movq	%rcx, 40(%rdi)
	movq	%rdx, 16(%rdi)
	movq	$1, 24(%rdi)
	retq
.Lfunc_end4:
	.size	_ZN4core3fmt9Arguments6new_v117heae077875e0c9aa4E, .Lfunc_end4-_ZN4core3fmt9Arguments6new_v117heae077875e0c9aa4E
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE
	.type	_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE,@function
_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE: # @_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE
	.cfi_startproc
# %bb.0:                                # %start
	movq	%rdi, %rax
	movq	%rsi, (%rdi)
	movq	$1, 8(%rdi)
	movq	.L__unnamed_1, %rdx
	movq	.L__unnamed_1+8, %rcx
	movq	%rdx, 32(%rdi)
	movq	%rcx, 40(%rdi)
	movl	$8, %ecx
	movq	%rcx, 16(%rdi)
	movq	$0, 24(%rdi)
	retq
.Lfunc_end5:
	.size	_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE, .Lfunc_end5-_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hc1bec46d3b8395e5E
	.type	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hc1bec46d3b8395e5E,@function
_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hc1bec46d3b8395e5E: # @"_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hc1bec46d3b8395e5E"
	.cfi_startproc
# %bb.0:                                # %start
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	(%rdi), %rdi
	callq	_ZN4core3ops8function6FnOnce9call_once17h064fc3f91b2bf0efE
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end6:
	.size	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hc1bec46d3b8395e5E, .Lfunc_end6-_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hc1bec46d3b8395e5E
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN4core3ops8function6FnOnce9call_once17h064fc3f91b2bf0efE
	.type	_ZN4core3ops8function6FnOnce9call_once17h064fc3f91b2bf0efE,@function
_ZN4core3ops8function6FnOnce9call_once17h064fc3f91b2bf0efE: # @_ZN4core3ops8function6FnOnce9call_once17h064fc3f91b2bf0efE
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 3, rust_eh_personality
	.cfi_lsda 3, .Lexception0
# %bb.0:                                # %start
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	%rdi, 8(%rsp)
.Ltmp0:
	leaq	8(%rsp), %rdi
	callq	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h433b39b893cf1495E
.Ltmp1:
	movl	%eax, 4(%rsp)                   # 4-byte Spill
	jmp	.LBB7_3
.LBB7_1:                                # %bb3
	movq	24(%rsp), %rdi
	callq	_Unwind_Resume@PLT
.LBB7_2:                                # %cleanup
.Ltmp2:
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, 24(%rsp)
	movl	%eax, 32(%rsp)
	jmp	.LBB7_1
.LBB7_3:                                # %bb1
	movl	4(%rsp), %eax                   # 4-byte Reload
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end7:
	.size	_ZN4core3ops8function6FnOnce9call_once17h064fc3f91b2bf0efE, .Lfunc_end7-_ZN4core3ops8function6FnOnce9call_once17h064fc3f91b2bf0efE
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table7:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 1 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Lfunc_end7-.Ltmp1             #   Call between .Ltmp1 and .Lfunc_end7
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _ZN4core3ops8function6FnOnce9call_once17h6f82b81b85a088f7E
	.type	_ZN4core3ops8function6FnOnce9call_once17h6f82b81b85a088f7E,@function
_ZN4core3ops8function6FnOnce9call_once17h6f82b81b85a088f7E: # @_ZN4core3ops8function6FnOnce9call_once17h6f82b81b85a088f7E
	.cfi_startproc
# %bb.0:                                # %start
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	*%rdi
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end8:
	.size	_ZN4core3ops8function6FnOnce9call_once17h6f82b81b85a088f7E, .Lfunc_end8-_ZN4core3ops8function6FnOnce9call_once17h6f82b81b85a088f7E
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN4core3ptr85drop_in_place$LT$std..rt..lang_start$LT$$LP$$RP$$GT$..$u7b$$u7b$closure$u7d$$u7d$$GT$17h54514d376286d1f0E
	.type	_ZN4core3ptr85drop_in_place$LT$std..rt..lang_start$LT$$LP$$RP$$GT$..$u7b$$u7b$closure$u7d$$u7d$$GT$17h54514d376286d1f0E,@function
_ZN4core3ptr85drop_in_place$LT$std..rt..lang_start$LT$$LP$$RP$$GT$..$u7b$$u7b$closure$u7d$$u7d$$GT$17h54514d376286d1f0E: # @"_ZN4core3ptr85drop_in_place$LT$std..rt..lang_start$LT$$LP$$RP$$GT$..$u7b$$u7b$closure$u7d$$u7d$$GT$17h54514d376286d1f0E"
	.cfi_startproc
# %bb.0:                                # %start
	retq
.Lfunc_end9:
	.size	_ZN4core3ptr85drop_in_place$LT$std..rt..lang_start$LT$$LP$$RP$$GT$..$u7b$$u7b$closure$u7d$$u7d$$GT$17h54514d376286d1f0E, .Lfunc_end9-_ZN4core3ptr85drop_in_place$LT$std..rt..lang_start$LT$$LP$$RP$$GT$..$u7b$$u7b$closure$u7d$$u7d$$GT$17h54514d376286d1f0E
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17h4b8822d538ca828fE
	.type	_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17h4b8822d538ca828fE,@function
_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17h4b8822d538ca828fE: # @"_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17h4b8822d538ca828fE"
	.cfi_startproc
# %bb.0:                                # %start
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	retq
.Lfunc_end10:
	.size	_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17h4b8822d538ca828fE, .Lfunc_end10-_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17h4b8822d538ca828fE
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN6linear4main17h7736007ed1673b53E
	.type	_ZN6linear4main17h7736007ed1673b53E,@function
_ZN6linear4main17h7736007ed1673b53E:    # @_ZN6linear4main17h7736007ed1673b53E
	.cfi_startproc
# %bb.0:                                # %start
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	leaq	8(%rsp), %rdi
	movabsq	$.Lalloc_537aa66a752b02fd65156d8ead2fb8dd, %rsi
	callq	_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE
	leaq	8(%rsp), %rdi
	callq	*_ZN3std2io5stdio6_print17ha3861c4b52105cd3E@GOTPCREL(%rip)
	callq	_ZN6linear1a17h50c02d9a9106acecE
	callq	_ZN6linear1d17h7cdf8599716e7201E
	leaq	56(%rsp), %rdi
	movabsq	$.Lalloc_a885df22687ea37f906b05324d617ad3, %rsi
	callq	_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE
	leaq	56(%rsp), %rdi
	callq	*_ZN3std2io5stdio6_print17ha3861c4b52105cd3E@GOTPCREL(%rip)
	addq	$104, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end11:
	.size	_ZN6linear4main17h7736007ed1673b53E, .Lfunc_end11-_ZN6linear4main17h7736007ed1673b53E
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN6linear1a17h50c02d9a9106acecE
	.type	_ZN6linear1a17h50c02d9a9106acecE,@function
_ZN6linear1a17h50c02d9a9106acecE:       # @_ZN6linear1a17h50c02d9a9106acecE
	.cfi_startproc
# %bb.0:                                # %start
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	leaq	16(%rsp), %rdi
	movabsq	$.Lalloc_77216ff829b12dd8466eaa6168a12586, %rsi
	callq	_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE
	leaq	16(%rsp), %rdi
	callq	*_ZN3std2io5stdio6_print17ha3861c4b52105cd3E@GOTPCREL(%rip)
	movl	$1, 68(%rsp)
	movl	68(%rsp), %eax
	incl	%eax
	movl	%eax, 12(%rsp)                  # 4-byte Spill
	seto	%al
	jo	.LBB12_2
# %bb.1:                                # %bb3
	movl	12(%rsp), %eax                  # 4-byte Reload
	movl	%eax, 68(%rsp)
	leaq	136(%rsp), %rdi
	leaq	68(%rsp), %rsi
	callq	_ZN4core3fmt2rt8Argument11new_display17h550d7fff055c1ad2E
	movq	136(%rsp), %rax
	movq	%rax, 120(%rsp)
	movq	144(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	72(%rsp), %rdi
	movabsq	$.Lalloc_9771be2481f51be410bd2ac520d18601, %rsi
	leaq	120(%rsp), %rdx
	callq	_ZN4core3fmt9Arguments6new_v117heae077875e0c9aa4E
	leaq	72(%rsp), %rdi
	callq	*_ZN3std2io5stdio6_print17ha3861c4b52105cd3E@GOTPCREL(%rip)
	callq	_ZN6linear1b17hf214847a4473eaffE
	addq	$152, %rsp
	.cfi_def_cfa_offset 8
	retq
.LBB12_2:                               # %panic
	.cfi_def_cfa_offset 160
	movabsq	$.Lalloc_6d7f7a5b84e017e6f8c6d7ddf7ec8e6f, %rdi
	callq	*_ZN4core9panicking11panic_const24panic_const_add_overflow17hca367bc6c4b9279fE@GOTPCREL(%rip)
.Lfunc_end12:
	.size	_ZN6linear1a17h50c02d9a9106acecE, .Lfunc_end12-_ZN6linear1a17h50c02d9a9106acecE
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN6linear1b17hf214847a4473eaffE
	.type	_ZN6linear1b17hf214847a4473eaffE,@function
_ZN6linear1b17hf214847a4473eaffE:       # @_ZN6linear1b17hf214847a4473eaffE
	.cfi_startproc
# %bb.0:                                # %start
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
	leaq	8(%rsp), %rdi
	movabsq	$.Lalloc_71de17250bd849c7513c2c122fb04cb0, %rsi
	callq	_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE
	leaq	8(%rsp), %rdi
	callq	*_ZN3std2io5stdio6_print17ha3861c4b52105cd3E@GOTPCREL(%rip)
	callq	_ZN6linear1c17h88729c0059217180E
	addq	$56, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end13:
	.size	_ZN6linear1b17hf214847a4473eaffE, .Lfunc_end13-_ZN6linear1b17hf214847a4473eaffE
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN6linear1c17h88729c0059217180E
	.type	_ZN6linear1c17h88729c0059217180E,@function
_ZN6linear1c17h88729c0059217180E:       # @_ZN6linear1c17h88729c0059217180E
	.cfi_startproc
# %bb.0:                                # %start
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
	leaq	8(%rsp), %rdi
	movabsq	$.Lalloc_06d14f6ce6044c51d8c355fe50b5048a, %rsi
	callq	_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE
	leaq	8(%rsp), %rdi
	callq	*_ZN3std2io5stdio6_print17ha3861c4b52105cd3E@GOTPCREL(%rip)
	addq	$56, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end14:
	.size	_ZN6linear1c17h88729c0059217180E, .Lfunc_end14-_ZN6linear1c17h88729c0059217180E
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZN6linear1d17h7cdf8599716e7201E
	.type	_ZN6linear1d17h7cdf8599716e7201E,@function
_ZN6linear1d17h7cdf8599716e7201E:       # @_ZN6linear1d17h7cdf8599716e7201E
	.cfi_startproc
# %bb.0:                                # %start
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
	leaq	8(%rsp), %rdi
	movabsq	$.Lalloc_79fc047f948285ece0c3ca1a3ed0d78e, %rsi
	callq	_ZN4core3fmt9Arguments9new_const17h0b478d4fbc455fffE
	leaq	8(%rsp), %rdi
	callq	*_ZN3std2io5stdio6_print17ha3861c4b52105cd3E@GOTPCREL(%rip)
	addq	$56, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end15:
	.size	_ZN6linear1d17h7cdf8599716e7201E, .Lfunc_end15-_ZN6linear1d17h7cdf8599716e7201E
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %top
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	%rsi, %rdx
	movslq	%edi, %rsi
	movabsq	$_ZN6linear4main17h7736007ed1673b53E, %rdi
	xorl	%ecx, %ecx
	callq	_ZN3std2rt10lang_start17hb92c0478cf91f2feE
                                        # kill: def $eax killed $eax killed $rax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end16:
	.size	main, .Lfunc_end16-main
	.cfi_endproc
                                        # -- End function
	.type	.Lvtable.0,@object              # @vtable.0
	.section	.rodata,"a",@progbits
	.p2align	3, 0x0
.Lvtable.0:
	.asciz	"\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hc1bec46d3b8395e5E
	.quad	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h433b39b893cf1495E
	.quad	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h433b39b893cf1495E
	.size	.Lvtable.0, 48

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	3, 0x0
.L__unnamed_1:
	.zero	8
	.zero	8
	.size	.L__unnamed_1, 16

	.type	.Lalloc_9fe35afcb1d958ac688308a0323eca4a,@object # @alloc_9fe35afcb1d958ac688308a0323eca4a
	.section	.rodata,"a",@progbits
.Lalloc_9fe35afcb1d958ac688308a0323eca4a:
	.ascii	"Starting...\n"
	.size	.Lalloc_9fe35afcb1d958ac688308a0323eca4a, 12

	.type	.Lalloc_537aa66a752b02fd65156d8ead2fb8dd,@object # @alloc_537aa66a752b02fd65156d8ead2fb8dd
	.p2align	3, 0x0
.Lalloc_537aa66a752b02fd65156d8ead2fb8dd:
	.quad	.Lalloc_9fe35afcb1d958ac688308a0323eca4a
	.asciz	"\f\000\000\000\000\000\000"
	.size	.Lalloc_537aa66a752b02fd65156d8ead2fb8dd, 16

	.type	.Lalloc_2e0d4f77a3e19c7f48ee1a5d1ef3ca2e,@object # @alloc_2e0d4f77a3e19c7f48ee1a5d1ef3ca2e
.Lalloc_2e0d4f77a3e19c7f48ee1a5d1ef3ca2e:
	.ascii	"Finished.\n"
	.size	.Lalloc_2e0d4f77a3e19c7f48ee1a5d1ef3ca2e, 10

	.type	.Lalloc_a885df22687ea37f906b05324d617ad3,@object # @alloc_a885df22687ea37f906b05324d617ad3
	.p2align	3, 0x0
.Lalloc_a885df22687ea37f906b05324d617ad3:
	.quad	.Lalloc_2e0d4f77a3e19c7f48ee1a5d1ef3ca2e
	.asciz	"\n\000\000\000\000\000\000"
	.size	.Lalloc_a885df22687ea37f906b05324d617ad3, 16

	.type	.Lalloc_083680540fac5c8a4ab975f01f059043,@object # @alloc_083680540fac5c8a4ab975f01f059043
.Lalloc_083680540fac5c8a4ab975f01f059043:
	.ascii	"In a\n"
	.size	.Lalloc_083680540fac5c8a4ab975f01f059043, 5

	.type	.Lalloc_77216ff829b12dd8466eaa6168a12586,@object # @alloc_77216ff829b12dd8466eaa6168a12586
	.p2align	3, 0x0
.Lalloc_77216ff829b12dd8466eaa6168a12586:
	.quad	.Lalloc_083680540fac5c8a4ab975f01f059043
	.asciz	"\005\000\000\000\000\000\000"
	.size	.Lalloc_77216ff829b12dd8466eaa6168a12586, 16

	.type	.Lalloc_343c077ece783e9f2cb2c65e53dc6c1c,@object # @alloc_343c077ece783e9f2cb2c65e53dc6c1c
.Lalloc_343c077ece783e9f2cb2c65e53dc6c1c:
	.ascii	"linear.rs"
	.size	.Lalloc_343c077ece783e9f2cb2c65e53dc6c1c, 9

	.type	.Lalloc_6d7f7a5b84e017e6f8c6d7ddf7ec8e6f,@object # @alloc_6d7f7a5b84e017e6f8c6d7ddf7ec8e6f
	.p2align	3, 0x0
.Lalloc_6d7f7a5b84e017e6f8c6d7ddf7ec8e6f:
	.quad	.Lalloc_343c077ece783e9f2cb2c65e53dc6c1c
	.asciz	"\t\000\000\000\000\000\000\000\013\000\000\000\t\000\000"
	.size	.Lalloc_6d7f7a5b84e017e6f8c6d7ddf7ec8e6f, 24

	.type	.Lalloc_49a1e817e911805af64bbc7efb390101,@object # @alloc_49a1e817e911805af64bbc7efb390101
.Lalloc_49a1e817e911805af64bbc7efb390101:
	.byte	10
	.size	.Lalloc_49a1e817e911805af64bbc7efb390101, 1

	.type	.Lalloc_9771be2481f51be410bd2ac520d18601,@object # @alloc_9771be2481f51be410bd2ac520d18601
	.p2align	3, 0x0
.Lalloc_9771be2481f51be410bd2ac520d18601:
	.quad	1
	.zero	8
	.quad	.Lalloc_49a1e817e911805af64bbc7efb390101
	.asciz	"\001\000\000\000\000\000\000"
	.size	.Lalloc_9771be2481f51be410bd2ac520d18601, 32

	.type	.Lalloc_ec96afa34c4660aaab8edc8f5dbf2a07,@object # @alloc_ec96afa34c4660aaab8edc8f5dbf2a07
.Lalloc_ec96afa34c4660aaab8edc8f5dbf2a07:
	.ascii	"In b\n"
	.size	.Lalloc_ec96afa34c4660aaab8edc8f5dbf2a07, 5

	.type	.Lalloc_71de17250bd849c7513c2c122fb04cb0,@object # @alloc_71de17250bd849c7513c2c122fb04cb0
	.p2align	3, 0x0
.Lalloc_71de17250bd849c7513c2c122fb04cb0:
	.quad	.Lalloc_ec96afa34c4660aaab8edc8f5dbf2a07
	.asciz	"\005\000\000\000\000\000\000"
	.size	.Lalloc_71de17250bd849c7513c2c122fb04cb0, 16

	.type	.Lalloc_f53bce841b8561e96c952a9cd7b68d8c,@object # @alloc_f53bce841b8561e96c952a9cd7b68d8c
.Lalloc_f53bce841b8561e96c952a9cd7b68d8c:
	.ascii	"In c\n"
	.size	.Lalloc_f53bce841b8561e96c952a9cd7b68d8c, 5

	.type	.Lalloc_06d14f6ce6044c51d8c355fe50b5048a,@object # @alloc_06d14f6ce6044c51d8c355fe50b5048a
	.p2align	3, 0x0
.Lalloc_06d14f6ce6044c51d8c355fe50b5048a:
	.quad	.Lalloc_f53bce841b8561e96c952a9cd7b68d8c
	.asciz	"\005\000\000\000\000\000\000"
	.size	.Lalloc_06d14f6ce6044c51d8c355fe50b5048a, 16

	.type	.Lalloc_0f26088e69b8f0dab93505c68a0dfe1e,@object # @alloc_0f26088e69b8f0dab93505c68a0dfe1e
.Lalloc_0f26088e69b8f0dab93505c68a0dfe1e:
	.ascii	"In d\n"
	.size	.Lalloc_0f26088e69b8f0dab93505c68a0dfe1e, 5

	.type	.Lalloc_79fc047f948285ece0c3ca1a3ed0d78e,@object # @alloc_79fc047f948285ece0c3ca1a3ed0d78e
	.p2align	3, 0x0
.Lalloc_79fc047f948285ece0c3ca1a3ed0d78e:
	.quad	.Lalloc_0f26088e69b8f0dab93505c68a0dfe1e
	.asciz	"\005\000\000\000\000\000\000"
	.size	.Lalloc_79fc047f948285ece0c3ca1a3ed0d78e, 16

	.ident	"rustc version 1.86.0 (05f9846f8 2025-03-31)"
	.section	".note.GNU-stack","",@progbits
