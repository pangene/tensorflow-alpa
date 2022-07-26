#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_REPLICA_GROUP_RESOLVER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_REPLICA_GROUP_RESOLVER_H_
 
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
 
namespace xla {
namespace spmd {
 
// Converts replica_groups={{0}} to replica_groups={}. EXPERIMENTAL
//
// Before:
// %all-reduce.14 = f16[16,512]{1,0} all-reduce(f16[16,512]{1,0} %reduce.7), channel_id=1, replica_groups={{0}}, to_apply=%primitive_computation_add__19.932
//
// After:
// %all-reduce.14 = f16[16,512]{1,0} all-reduce(f16[16,512]{1,0} %reduce.7), channel_id=1, replica_groups={}, to_apply=%primitive_computation_add__19.932
//

class ReplicaGroupResolver : public HloModulePass {
 public:
  ReplicaGroupResolver() = default;
  ~ReplicaGroupResolver() override = default;
  absl::string_view name() const override {
    return "replica_group_resolver";
  }
 
  StatusOr<bool> Run(HloModule* module) override;
};
 
}  // namespace spmd
}  // namespace xla
 
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_REPLICA_GROUP_RESOLVER_H_