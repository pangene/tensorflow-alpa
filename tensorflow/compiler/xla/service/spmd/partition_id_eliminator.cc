#include "tensorflow/compiler/xla/service/spmd/partition_id_eliminator.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
namespace spmd {

using absl::StrCat;

StatusOr<bool> PartitionIdEliminator::Run(HloModule* module) {
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* ins : computation->instructions()) {
      if (ins->opcode() == HloOpcode::kPartitionId) {
        int param_num = ins->parent()->num_parameters();
        HloInstruction* param_ins = ins->parent()->AddEntryComputationParameter(
            HloInstruction::CreateParameter(param_num, ins->shape(), StrCat("param_", param_num))
        );
        ins->ReplaceAllUsesWith(param_ins);
        changed = true;
      }
    }
  }

  if (changed) {
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  }

  return changed;
}

}  // namespace spmd
}  // namespace xla
